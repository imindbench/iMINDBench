"""
Shared PyTorch model base class for imindbench models.
"""

from abc import abstractmethod

import numpy as np
import torch
from omegaconf import DictConfig

from imindbench.utils.logging_utils import log

from .base_model import BaseModel


class TorchBaseModel(BaseModel):
    """Shared torch-model wrapper used by the neural baseline implementations."""

    def __init__(self, cfg: DictConfig, dataset_cfg: DictConfig | None = None):
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.device = None
        self.model = None
        self.random_state = cfg.get("random_state", 42)
        # TorchRunner seeds each fold before building the network.
        # Model construction must not override that seed.

    @staticmethod
    def _get_device(cfg):
        """Get device from config or auto-detect."""
        device_str = cfg.get("device", "auto")

        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_str == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            device = torch.device("cuda")
        elif device_str == "cpu":
            device = torch.device("cpu")
        else:
            # Allow specific device strings like 'cuda:0'
            device = torch.device(device_str)

        # Verify device is valid and log
        if device.type == "cuda":
            if device.index is not None:
                if device.index >= torch.cuda.device_count():
                    import warnings

                    warnings.warn(
                        f"GPU {device.index} not available (only {torch.cuda.device_count()} GPUs). Falling back to CPU.",
                        stacklevel=2,
                    )
                    device = torch.device("cpu")
                else:
                    # Set the current device to ensure it's accessible
                    torch.cuda.set_device(device.index)

        log(
            f"[TorchBaseModel] Device configured: {device_str} -> {device} (CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0})",
            priority=2,
        )
        return device

    def _resolve_device(self, device=None):
        """Resolve model device from runner override or model config."""
        resolved = (
            self._get_device(self.cfg) if device is None else torch.device(device)
        )
        self.device = resolved
        return resolved

    @abstractmethod
    def _create_network(self, input_shape, n_classes):
        """Create the neural network architecture."""
        pass

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fit is implemented by trainers, but models need this for interface."""
        raise NotImplementedError("PyTorch models should be trained via trainers")

    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model is None:
            raise RuntimeError("Model must be built before calling predict_proba.")
        if self.device is None:
            self._resolve_device()
        self.model.eval()
        all_probs = []
        batch_size = self.cfg.get("batch_size", 64)

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i : i + batch_size].to(self.device)
                outputs = self.model(batch_X)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def get_device(self):
        """Get the device (CPU or CUDA)."""
        if self.device is None:
            self._resolve_device()
        return self.device

    def prepare_batch(self, batch, **kwargs):
        """Default torch-model batch preparation hook with basic validation."""
        _ = kwargs
        if not isinstance(batch, dict):
            raise TypeError(f"prepare_batch expected dict, got {type(batch).__name__}.")
        if "x" not in batch or "y" not in batch:
            raise KeyError("prepare_batch expects keys 'x' and 'y'.")
        return batch
