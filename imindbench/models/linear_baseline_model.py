"""
Linear baseline model from PopT-BYD-BTB.
Single linear layer with optional dropout, batch normalization, and sensor dropout.
"""

import numpy as np
import torch
import torch.nn as nn

from . import register_model
from .torch_base_model import TorchBaseModel


@register_model("linear_baseline")
class LinearBaselineModel(TorchBaseModel):
    """
    Linear baseline model from PopT-BYD-BTB.

    Differences from PopT-BYD-BTB:
    - Outputs n_classes instead of 1 (for multi-class neuroprobe tasks)
    - Inherits from neuroprobe's TorchBaseModel instead of PopT's BaseModel
    - build_model signature: build_model(input_shape, n_classes) instead of build_model(cfg)

    Features:
    - Optional batch normalization
    - Dropout layer
    - Sensor dropout (training only, requires >2 electrodes)
    - Input flattening: inputs.flatten(start_dim=1)
    """

    def _create_network(
        self, input_dim, n_classes, dropout_p, sensor_dropout_p, use_batch_norm
    ):
        """Create the linear network architecture."""

        class LinearBaseline(nn.Module):
            def __init__(
                self, input_dim, n_classes, dropout_p, sensor_dropout_p, use_batch_norm
            ):
                super().__init__()
                self.input_dim = input_dim
                self.n_classes = n_classes
                self.dropout_p = dropout_p
                self.sensor_dropout_p = sensor_dropout_p
                self.use_batch_norm = use_batch_norm

                if use_batch_norm:
                    self.batch_norm = nn.BatchNorm1d(input_dim)
                self.dropout = nn.Dropout(p=dropout_p)
                self.linear_out = nn.Linear(
                    in_features=input_dim, out_features=n_classes
                )

            def forward(self, inputs):
                """
                Forward pass with sensor dropout, batch norm, dropout, and linear layer.

                Args:
                    inputs: Input tensor of shape (batch_size, ...) - will be flattened

                Returns:
                    Output tensor of shape (batch_size, n_classes)
                """
                if self.training and self.sensor_dropout_p > 0:
                    # Sensor dropout only makes sense before flattening when an
                    # explicit channel axis is still present.
                    if inputs.ndim >= 2 and inputs.size(1) > 2:
                        apply_mask = (
                            torch.rand(inputs.size(1), device=inputs.device)
                            < self.sensor_dropout_p
                        )
                        inputs = inputs.clone()
                        inputs[:, apply_mask, :] = torch.zeros(
                            inputs[:, apply_mask, :].size(), device=inputs.device
                        )

                flattened_input = inputs.flatten(start_dim=1)
                if self.use_batch_norm:
                    flattened_input = self.batch_norm(flattened_input)
                out = self.dropout(flattened_input)
                out = self.linear_out(out)
                return out

        return LinearBaseline(
            input_dim, n_classes, dropout_p, sensor_dropout_p, use_batch_norm
        )

    def build_model(self, input_shape, n_classes, device=None):
        """
        Build the model with given input shape and number of classes.

        Args:
            input_shape: Input shape tuple (will be flattened)
            n_classes: Number of output classes
        """
        self._resolve_device(device)
        input_dim = np.prod(input_shape)

        dropout_p = self.cfg.get("dropout", 0.0)
        sensor_dropout_p = self.cfg.get("sensor_dropout", 0.0)
        use_batch_norm = self.cfg.get("batch_norm", False)

        self.model = self._create_network(
            input_dim, n_classes, dropout_p, sensor_dropout_p, use_batch_norm
        )
        self.model = self.model.to(self.device)
        self.classes_ = np.arange(n_classes)
        return self.model
