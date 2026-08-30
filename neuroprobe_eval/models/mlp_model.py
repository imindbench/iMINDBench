"""
Multi-Layer Perceptron model (PyTorch).
"""

import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
from . import register_model
from .model_utils import to_int_list
from .torch_base_model import TorchBaseModel


@register_model("mlp")
class MLPModel(TorchBaseModel):
    """Multi-Layer Perceptron classifier."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.hidden_dims = to_int_list(
            cfg.get("hidden_dims", [1024, 1024]),
            [1024, 1024],
            allow_empty=True,
        )

    def _create_network(self, input_size, n_classes):
        """Create the feed-forward network used by the classifier head."""

        class MLP(nn.Module):
            def __init__(self, input_size, n_classes, hidden_dims):
                super().__init__()
                layers = []

                if len(hidden_dims) == 0:
                    # Linear model (logistic regression)
                    layers.append(nn.Linear(input_size, n_classes))
                else:
                    # MLP with hidden layers
                    prev_dim = input_size
                    for hidden_dim in hidden_dims:
                        layers.append(nn.Linear(prev_dim, hidden_dim))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(0.2))
                        prev_dim = hidden_dim

                    # Output layer
                    layers.append(nn.Linear(prev_dim, n_classes))

                self.network = nn.Sequential(*layers)

            def forward(self, x):
                # MLP variants always consume a flat per-sample feature vector.
                x = x.view(x.size(0), -1)
                return self.network(x)

        return MLP(input_size, n_classes, self.hidden_dims)

    def build_model(self, input_shape, n_classes, device=None):
        """Build the model with given input shape and number of classes."""
        self._resolve_device(device)
        input_size = np.prod(input_shape)
        self.model = self._create_network(input_size, n_classes)
        self.model = self.model.to(self.device)
        self.classes_ = np.arange(n_classes)
        return self.model
