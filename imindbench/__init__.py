"""iMINDBench evaluation package."""

from pathlib import Path

from omegaconf import OmegaConf

# Keep packaged config paths independent of the caller's working directory.
OmegaConf.register_new_resolver(
    "imindbench_path",
    lambda relative: str(Path(__file__).resolve().parent / relative),
    replace=True,
)
