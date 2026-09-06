import numpy as np
import torch
from scipy.signal import resample_poly

from . import register_preprocessor
from .base_preprocessor import BasePreprocessor


@register_preprocessor("downsample")
class DownsamplePreprocessor(BasePreprocessor):
    """Preprocessor that downsamples neural data."""

    def _transform_one(self, sample):
        if not isinstance(sample, dict):
            raise TypeError(f"Expected sample dict, got {type(sample).__name__}.")
        if "x" not in sample:
            raise KeyError("downsample preprocessor requires sample['x'].")

        source_rate = self.cfg.get("source_rate", 2048)
        target_rate = self.cfg.get("target_rate", 500)

        out = dict(sample)
        x = sample["x"]
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)

        from math import gcd

        g = gcd(int(source_rate), int(target_rate))
        up = int(target_rate // g)
        down = int(source_rate // g)

        x_down = resample_poly(
            x_np,
            up=up,
            down=down,
            axis=-1,
        ).astype(np.float32, copy=False)

        out["x"] = x_down

        return out

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]
