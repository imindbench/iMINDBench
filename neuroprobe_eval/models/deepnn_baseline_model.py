"""
DeepNN baseline model from PopT-BYD-BTB.
4-layer deep neural network with 512 hidden units each and GELU activations.
"""

import torch.nn as nn
import numpy as np
from . import register_model
from .model_utils import coerce_int_list
from .torch_base_model import TorchBaseModel


@register_model("deepnn_baseline")
class DeepNNBaselineModel(TorchBaseModel):
    """
    DeepNN baseline model from PopT-BYD-BTB.

    Differences from PopT-BYD-BTB:
    - Outputs n_classes instead of 1 (for multi-class neuroprobe tasks)
    - Inherits from neuroprobe's TorchBaseModel instead of PopT's BaseModel
    - build_model signature: build_model(input_shape, n_classes) instead of build_model(cfg)

    Architecture:
    - 4 hidden layers with 512 units each
    - GELU activations
    - Optional batch normalization on input
    - Input flattening: inputs.flatten(start_dim=1)
    """

    def _create_network(self, input_dim, n_classes, use_batch_norm, hidden_dims):
        """Create the deep neural network architecture."""

        class DeepNNBaseline(nn.Module):
            def __init__(self, input_dim, n_classes, use_batch_norm, hidden_dims):
                super().__init__()
                self.input_dim = input_dim
                self.n_classes = n_classes
                self.use_batch_norm = use_batch_norm

                if use_batch_norm:
                    self.batch_norm = nn.BatchNorm1d(input_dim)

                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.append(
                        nn.Linear(in_features=prev_dim, out_features=hidden_dim)
                    )
                    layers.append(nn.GELU())
                    prev_dim = hidden_dim
                layers.append(nn.Linear(in_features=prev_dim, out_features=n_classes))
                self.network = nn.Sequential(*layers)

            def forward(self, inputs):
                """
                Forward pass through the deep network.

                Args:
                    inputs: Input tensor of shape (batch_size, ...) - will be flattened

                Returns:
                    Output tensor of shape (batch_size, n_classes)
                """
                flattened_input = inputs.flatten(start_dim=1)
                if self.use_batch_norm:
                    flattened_input = self.batch_norm(flattened_input)
                return self.network(flattened_input)

        return DeepNNBaseline(input_dim, n_classes, use_batch_norm, hidden_dims)

    def build_model(self, input_shape, n_classes, device=None):
        """
        Build the model with given input shape and number of classes.

        Args:
            input_shape: Input shape tuple (will be flattened)
            n_classes: Number of output classes
        """
        self._resolve_device(device)
        input_dim = np.prod(input_shape)
        use_batch_norm = self.cfg.get("batch_norm", False)
        self.hidden_dims = coerce_int_list(
            self.cfg.get("hidden_dims", [512, 512, 512, 512]),
            [512, 512, 512, 512],
            allow_empty=True,
        )

        self.model = self._create_network(
            input_dim,
            n_classes,
            use_batch_norm,
            self.hidden_dims,
        )
        self.model = self.model.to(self.device)
        self.classes_ = np.arange(n_classes)
        return self.model
