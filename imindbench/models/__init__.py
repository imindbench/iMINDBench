"""
Model registry for automatic model discovery and instantiation.
"""

import importlib
import os
from pathlib import Path
from omegaconf import DictConfig

MODEL_REGISTRY = {}

# Model modules skipped at import time because a third-party dependency they
# need is not installed, as {module name: missing distribution}. Asking for one
# of these models raises with the missing dependency named; every other model
# stays usable.
UNAVAILABLE_MODEL_MODULES: dict[str, str] = {}


def register_model(name):
    """Decorator to register a model class."""

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"{name} already in registry")
        MODEL_REGISTRY[name] = cls
        return cls

    return register_model_cls


def build_model(cfg: DictConfig, dataset_cfg: DictConfig | None = None):
    """Build one registered model instance from config."""
    model_name = cfg.name
    if model_name not in MODEL_REGISTRY:
        missing = UNAVAILABLE_MODEL_MODULES.get(f"{model_name}_model")
        if missing is not None:
            raise ImportError(
                f"Model {model_name} needs {missing}, which is not installed. "
                f"Install {missing} using the repository-root environment.yml "
                "and public installation guidance, or choose another model."
            )
        raise ValueError(
            f"Model {model_name} not found in registry. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_name]
    if dataset_cfg is None:
        return model_class(cfg)
    return model_class(cfg, dataset_cfg)


def import_models():
    """Auto-import all model files to register them.

    A model that needs a third-party package nobody else needs (xformers, say)
    must not make this package unimportable when that package is absent -- that
    would take down every other model and anything that imports one. Such a
    module is recorded in UNAVAILABLE_MODEL_MODULES and skipped.

    Only a missing *third-party* module is tolerated. A module that fails for
    any other reason, including a missing imindbench import, is a real
    break and propagates.
    """
    models_dir = os.path.dirname(__file__)
    for file in sorted(os.listdir(models_dir)):
        if file.endswith(".py") and not file.startswith("_") and file != "__init__.py":
            module_name = str(Path(file).with_suffix(""))
            try:
                importlib.import_module(f"{__name__}.{module_name}")
            except ModuleNotFoundError as exc:
                missing = (exc.name or "").split(".")[0]
                if not missing or missing == __name__.split(".")[0]:
                    raise
                UNAVAILABLE_MODEL_MODULES[module_name] = missing


# Import all models to register them
import_models()
