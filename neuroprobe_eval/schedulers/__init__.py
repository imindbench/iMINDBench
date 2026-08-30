"""Scheduler registry helpers for torch training loops."""

__all__ = ["build_scheduler", "RampUp", "CosineAnnealing"]


def _load_ramp_up():
    try:
        from .ramp_up import RampUp
    except ModuleNotFoundError as exc:
        if exc.name != "warmup_scheduler":
            raise
        raise ModuleNotFoundError(
            "The ramp_up scheduler requires warmup_scheduler. Install the "
            "neuroprobe_eval environment from examples/neuroprobe_eval/"
            "environment.yml before using scheduler.name=ramp_up."
        ) from exc
    return RampUp


def __getattr__(name):
    if name == "RampUp":
        return _load_ramp_up()
    if name == "CosineAnnealing":
        from .cosine_annealing import CosineAnnealing

        return CosineAnnealing
    raise AttributeError(name)


def build_scheduler(cfg, optim):
    """Build the configured learning-rate scheduler, or return ``None``."""
    if cfg is None:
        return None

    name = cfg.get("name", None)
    if name is None or name == "none":
        return None

    if name == "ramp_up":
        RampUp = _load_ramp_up()
        return RampUp(cfg, optim)
    if name == "cosine_annealing":
        from .cosine_annealing import CosineAnnealing

        return CosineAnnealing(cfg, optim)
    raise ValueError(
        f"Scheduler name '{name}' not found. Supported: 'ramp_up', "
        "'cosine_annealing', 'none'"
    )
