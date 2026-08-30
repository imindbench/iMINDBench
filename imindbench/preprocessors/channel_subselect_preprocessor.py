"""
Preprocessor that drops channels not included in an allowed list.
"""

import json
import os
import re

from .base_preprocessor import BasePreprocessor
from . import register_preprocessor
from .preprocessor_utils import (
    project_channel_metadata,
    validate_named_channel_sample,
)


def _normalize_labels_to_str(labels):
    """Normalize an allowlist to non-empty trimmed string labels."""
    if labels is None:
        return None
    out = []
    for idx, label in enumerate(labels):
        text = str(label).strip()
        if not text:
            raise ValueError(
                f"Allowed electrode label cannot be empty at index {idx}: {label!r}"
            )
        out.append(text)
    return out


@register_preprocessor("channel_subselect")
class ChannelSubselectPreprocessor(BasePreprocessor):
    """Selects a subset of channels based on allowed electrode labels."""

    def __init__(self, cfg):
        super().__init__(cfg)
        allowed = cfg.get("allowed_electrodes")
        max_channels = cfg.get("max_channels")
        if max_channels is not None:
            max_channels = int(max_channels)
            if max_channels < 1:
                raise ValueError(
                    f"channel_subselect max_channels must be positive, got {max_channels}."
                )
        self.max_channels = max_channels
        self.subject = cfg.get("subject")
        self.allowed_electrodes_by_subject = None
        self.allowed_electrodes = self._normalize_allowed(allowed)
        self._initial_allowed = (
            None if self.allowed_electrodes is None else set(self.allowed_electrodes)
        )
        self._initial_allowed_by_subject = self._copy_allowed_by_subject(
            self.allowed_electrodes_by_subject
        )

    def set_allowed_electrodes(self, electrode_labels):
        if electrode_labels is None:
            self._restore_initial_allowed()
        else:
            self.allowed_electrodes_by_subject = None
            self.allowed_electrodes = set(electrode_labels)

    def reset_state(self):
        self._restore_initial_allowed()

    def _restore_initial_allowed(self):
        self.allowed_electrodes = (
            None if self._initial_allowed is None else set(self._initial_allowed)
        )
        self.allowed_electrodes_by_subject = self._copy_allowed_by_subject(
            self._initial_allowed_by_subject
        )

    def _copy_allowed_by_subject(self, allowed_by_subject):
        """Deep-copy the subject->allowed-label mapping into mutable sets."""
        if allowed_by_subject is None:
            return None
        return {str(key): set(values) for key, values in allowed_by_subject.items()}

    def _normalize_subject_key(self, value):
        """Normalize an optional subject selector to a trimmed key or None."""
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None

    def _infer_subject_from_label(self, label):
        """Infer subject key from prefixed channel labels when possible."""
        if not isinstance(label, str) or "/" not in label:
            return None
        if not self.allowed_electrodes_by_subject:
            return None
        parts = [p.strip() for p in label.split("/") if p.strip()]
        for part in parts:
            if part in self.allowed_electrodes_by_subject:
                return part
        for part in parts:
            match = re.search(r"(sub_\d+)", part)
            if match:
                subject = match.group(1)
                if subject in self.allowed_electrodes_by_subject:
                    return subject
        return None

    def _is_allowed_label(self, label):
        """Return whether a channel label survives the configured allowlist."""
        if self.allowed_electrodes is None:
            return True
        inferred_subject = None
        suffix = None
        if isinstance(label, str) and "/" in label:
            suffix = label.split("/")[-1]
            inferred_subject = self._infer_subject_from_label(label)
            if inferred_subject is not None:
                # Subject-specific allowlists take precedence over any global
                # fallback/union when a subject prefix is present.
                subject_allowed = self.allowed_electrodes_by_subject.get(
                    inferred_subject, set()
                )
                if label in subject_allowed or suffix in subject_allowed:
                    return True
                return False
        if label in self.allowed_electrodes:
            return True
        if suffix is not None:
            # Variable-channel IDs may be prefixed as subject/session/channel.
            # Accept by raw channel-id suffix when clean lists are unprefixed.
            if suffix in self.allowed_electrodes:
                return True
        return False

    def _normalize_allowed(self, allowed):
        """Resolve allowed_electrodes from config value or JSON file."""
        if allowed is None:
            return None
        if isinstance(allowed, str):
            if not os.path.exists(allowed):
                raise FileNotFoundError(f"Allowed electrodes file not found: {allowed}")
            with open(allowed, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.allowed_electrodes_by_subject = {
                    str(key): set(values) for key, values in loaded.items()
                }
                subject_key = self._normalize_subject_key(self.subject)
                if subject_key is not None:
                    if subject_key in loaded:
                        loaded = loaded[subject_key]
                    else:
                        raise ValueError(
                            f"Subject {subject_key} not found in allowed electrodes JSON."
                        )
                else:
                    # No explicit subject selected: use union as default fallback and
                    # resolve subject-specific lists dynamically when prefixes exist.
                    merged = set()
                    for values in loaded.values():
                        merged.update(values)
                    loaded = list(merged)
            allowed = loaded
        return set(_normalize_labels_to_str(allowed))

    def _transform_one(self, sample):
        """Filter one sample by channel_names while preserving channel_ids alignment."""
        x_in, channel_ids_in, channel_names_in = validate_named_channel_sample(
            sample,
            context="channel_subselect",
        )

        if self.allowed_electrodes is None:
            aligned_old_indices = list(range(len(channel_names_in)))
        else:
            keep_mask = [self._is_allowed_label(name) for name in channel_names_in]
            aligned_old_indices = [i for i, keep in enumerate(keep_mask) if keep]
            if not aligned_old_indices:
                raise ValueError(
                    "No electrodes remain after channel_subselect preprocessing."
                )

        if (
            self.max_channels is not None
            and len(aligned_old_indices) > self.max_channels
        ):
            aligned_old_indices = aligned_old_indices[: self.max_channels]

        x_out_np = x_in[aligned_old_indices]
        channel_names_out = [channel_names_in[i] for i in aligned_old_indices]
        return project_channel_metadata(
            sample,
            context="channel_subselect",
            channel_ids_in=channel_ids_in,
            channel_names_in=channel_names_in,
            aligned_old_indices=aligned_old_indices,
            x_out=x_out_np,
            channel_names_out=channel_names_out,
        )

    def transform_samples(self, samples):
        """Apply channel filtering to every sample in the iterable."""
        return [self._transform_one(sample) for sample in samples]
