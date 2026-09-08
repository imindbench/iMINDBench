"""
Preprocessor registry for automatic preprocessor discovery and instantiation.
"""

import importlib
import math
import os
from copy import deepcopy
from pathlib import Path

from omegaconf import DictConfig, ListConfig

from .base_preprocessor import BasePreprocessor

PREPROCESSOR_REGISTRY = {}


def register_preprocessor(name):
    """Decorator to register a preprocessor class."""

    def register_preprocessor_cls(cls):
        if name in PREPROCESSOR_REGISTRY:
            raise ValueError(f"{name} already in registry")
        PREPROCESSOR_REGISTRY[name] = cls
        return cls

    return register_preprocessor_cls


class CompositePreprocessor(BasePreprocessor):
    """Run a sequence of preprocessors as one staged pipeline."""

    def __init__(self, cfg, preprocessors):
        super().__init__(cfg)
        self.preprocessors = preprocessors
        # Cache of fully transformed train samples produced during fit_split.
        # This lets variable-channel fold builders reuse train preprocessing work
        # instead of re-running the same chain during train materialization.
        self._fit_transformed_samples = None

    def set_allowed_electrodes(self, electrode_labels):
        """Forward optional electrode-selection hints to compatible stages."""
        for pre in self.preprocessors:
            if hasattr(pre, "set_allowed_electrodes"):
                pre.set_allowed_electrodes(electrode_labels)

    def reset_state(self):
        """Reset state for all preprocessors in the chain."""
        self._fit_transformed_samples = None
        for pre in self.preprocessors:
            if hasattr(pre, "reset_state"):
                pre.reset_state()

    def unload_model(self):
        """Forward model-unload hooks to all stages in the chain."""
        for pre in self.preprocessors:
            unload = getattr(pre, "unload_model", None)
            if callable(unload):
                unload()

    def requires_fit(self) -> bool:
        """Whether any stage needs a split-level fit pass before transform."""
        return any(pre.requires_fit() for pre in self.preprocessors)

    @property
    def invalidates_raw_channel_indices(self) -> bool:
        """Whether any child stage makes raw channel indices invalid."""
        return any(
            bool(getattr(pre, "invalidates_raw_channel_indices", False))
            for pre in self.preprocessors
        )

    def set_fold_context(self, context):
        """Forward fold-level metadata to stages that consume it."""
        effective_context = dict(context or {})
        alignment_samples = 1
        for pre in self.preprocessors:
            get_alignment = getattr(pre, "get_context_alignment_samples", None)
            if callable(get_alignment):
                value = get_alignment()
                if value is not None:
                    value = int(value)
                    if value <= 0:
                        raise ValueError(
                            "Context alignment samples must be positive, got "
                            f"{value} from {pre.__class__.__name__}."
                        )
                    alignment_samples = math.lcm(alignment_samples, value)
        effective_context["context_alignment_samples"] = alignment_samples
        for pre in self.preprocessors:
            if hasattr(pre, "set_fold_context"):
                pre.set_fold_context(effective_context)

    def fit_split(self, sample_iter):
        """Fit each stage in order and cache the transformed train samples."""
        # Materialize once so fit preprocessors later in the chain can consume
        # outputs from earlier transforms without re-reading a one-shot iterator.
        working_samples = [deepcopy(sample) for sample in sample_iter]
        states = []
        for pre in self.preprocessors:
            state = None
            if pre.requires_fit():
                state = pre.fit_split(iter(working_samples))
                pre.set_state(state)
            states.append(pre.get_state() if state is None else state)
            # Run per-stage sample transforms through the batch-capable hook so
            # heavy stages (for example encoders) can process multiple samples
            # in one call while preserving the canonical sample-dict contract.
            working_samples = pre.transform_samples(working_samples)
        self._fit_transformed_samples = working_samples
        return states

    def consume_fit_transformed_samples(self):
        """Return and clear cached transformed train samples from fit_split."""
        samples = self._fit_transformed_samples
        self._fit_transformed_samples = None
        return samples

    def set_state(self, state):
        """Restore stage states produced by a previous fit_split call."""
        if state is None:
            for pre in self.preprocessors:
                pre.set_state(None)
            return

        if not isinstance(state, (list, tuple)):
            raise TypeError("CompositePreprocessor state must be a list/tuple.")
        if len(state) != len(self.preprocessors):
            raise ValueError(
                "CompositePreprocessor state length mismatch: "
                f"expected {len(self.preprocessors)}, got {len(state)}."
            )

        for pre, pre_state in zip(self.preprocessors, state, strict=False):
            pre.set_state(pre_state)

    def get_state(self):
        """Return per-stage fitted state in pipeline order."""
        return [pre.get_state() for pre in self.preprocessors]

    def transform_samples(self, samples):
        """Run sample transforms stage-by-stage in pipeline order."""
        out_samples = list(samples)
        for pre in self.preprocessors:
            out_samples = pre.transform_samples(out_samples)
        return out_samples


def _build_single(cfg: DictConfig):
    """Build one registered preprocessor instance from config."""
    preprocessor_name = cfg.name
    if preprocessor_name not in PREPROCESSOR_REGISTRY:
        raise ValueError(
            f"Preprocessor {preprocessor_name} not found in registry. "
            f"Available: {list(PREPROCESSOR_REGISTRY.keys())}"
        )

    preprocessor_class = PREPROCESSOR_REGISTRY[preprocessor_name]
    return preprocessor_class(cfg)


def _iter_preprocessor_stages(preprocessors):
    for pre in preprocessors:
        if isinstance(pre, CompositePreprocessor):
            yield from _iter_preprocessor_stages(pre.preprocessors)
        else:
            yield pre


def _validate_context_window_order(preprocessors):
    stages = list(_iter_preprocessor_stages(preprocessors))
    seen_derived_channels = False
    seen_context_window = False
    seen_context_crop = False
    # Whether the context_window stage seen so far requires a downstream
    # crop_to_target_window stage. Defaults to True (today's behavior) and is
    # only relaxed when that stage was configured with crop_back=False, i.e.
    # it intentionally feeds the full context window straight to the model.
    context_window_requires_crop = True
    for idx, pre in enumerate(stages):
        if getattr(pre, "is_context_window_stage", False):
            seen_context_window = True
            context_window_requires_crop = bool(getattr(pre, "crop_back", True))
        if seen_context_window and getattr(
            pre, "is_crop_to_target_window_stage", False
        ):
            seen_context_crop = True
        if getattr(pre, "is_context_window_stage", False) and idx != 0:
            raise ValueError(
                "context_window must be the first stage in a preprocessor chain."
            )
        if getattr(pre, "is_context_window_stage", False) and seen_derived_channels:
            raise ValueError(
                "context_window must run before channel-changing or "
                "derived-channel stages."
            )
        if (
            seen_context_window
            and context_window_requires_crop
            and not seen_context_crop
            and getattr(pre, "is_standardization_stage", False)
        ):
            raise ValueError(
                "crop_to_target_window must run before standardize when using "
                "context_window."
            )
        if getattr(pre, "invalidates_raw_channel_indices", False):
            seen_derived_channels = True
    if seen_context_window and context_window_requires_crop and not seen_context_crop:
        raise ValueError(
            "context_window chains must include crop_to_target_window before "
            "downstream sample-level consumers."
        )


def describe_preprocessor(cfg):
    """Describe a validated pipeline by its ordered stage names, not a preset alias."""
    if isinstance(cfg, (ListConfig, list)):
        return " -> ".join(describe_preprocessor(stage) for stage in cfg)
    if "chain" in cfg:
        return describe_preprocessor(cfg["chain"])
    return cfg["name"]


def build_preprocessor(cfg):
    """Build a preprocessor or chain of preprocessors from configuration."""
    if isinstance(cfg, ListConfig) or isinstance(cfg, list):
        preprocessors = [build_preprocessor(stage_cfg) for stage_cfg in cfg]
        _validate_context_window_order(preprocessors)
        return CompositePreprocessor(cfg, preprocessors)

    chain_cfgs = cfg.get("chain") if hasattr(cfg, "get") else None
    if chain_cfgs:
        preprocessors = [build_preprocessor(stage_cfg) for stage_cfg in chain_cfgs]
        _validate_context_window_order(preprocessors)
        return CompositePreprocessor(cfg, preprocessors)

    return _build_single(cfg)


def import_preprocessors():
    """Auto-import all preprocessor files to register them."""
    preprocessors_dir = os.path.dirname(__file__)
    for file in os.listdir(preprocessors_dir):
        if (
            file.endswith(".py")
            and not file.startswith("_")
            and file != "base_preprocessor.py"
        ):
            module_name = str(Path(file).with_suffix(""))
            importlib.import_module(f"{__name__}.{module_name}")


# Import all preprocessors to register them
import_preprocessors()
