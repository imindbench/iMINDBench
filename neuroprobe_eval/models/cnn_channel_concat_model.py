"""
Channel-concat CNN model (PyTorch).

Applies a shared 2D CNN trunk independently to each channel's STFT, then
concatenates per-channel embeddings before the final classifier head.
"""

import numpy as np
import torch
import torch.nn as nn

from . import register_model
from .cnn_model import prepare_cnn_batch
from .model_utils import coerce_int_list
from .torch_base_model import TorchBaseModel


@register_model("cnn_channel_concat")
class CNNChannelConcatModel(TorchBaseModel):
    """CNN classifier with per-channel trunk and late channel concatenation."""

    def prepare_batch(self, batch, **kwargs):
        """Prepare collated batch for channel-concat CNN input convention."""
        return prepare_cnn_batch(batch, **kwargs)

    def _create_network(
        self,
        input_shape,
        n_classes,
        hidden_dims,
        conv_channels,
        kernel_size,
        pool_kernel,
        dropout,
        channel_batch_size,
    ):
        class ChannelConcatCNN(nn.Module):
            def __init__(
                self,
                input_shape,
                n_classes,
                hidden_dims,
                conv_channels,
                kernel_size,
                pool_kernel,
                dropout,
                channel_batch_size,
            ):
                super().__init__()
                if len(input_shape) != 3:
                    raise ValueError(
                        "cnn_channel_concat expects input_shape=(channels, freqs, times), "
                        f"got {input_shape}."
                    )

                n_channels, n_freqs, n_times = input_shape
                if n_channels < 1 or n_freqs < 1 or n_times < 1:
                    raise ValueError(
                        f"Invalid input_shape for cnn_channel_concat: {input_shape}"
                    )

                self.n_channels = int(n_channels)
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool2d(pool_kernel)
                self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.dropout = nn.Dropout(dropout)
                self.channel_batch_size = int(channel_batch_size)
                if self.channel_batch_size <= 0:
                    self.channel_batch_size = int(n_channels)

                # Shared per-channel trunk: in_channels=1 is intentional.
                convs = []
                in_ch = 1
                padding = kernel_size // 2
                for out_ch in conv_channels:
                    convs.append(
                        nn.Conv2d(
                            in_ch, out_ch, kernel_size=kernel_size, padding=padding
                        )
                    )
                    in_ch = out_ch
                self.convs = nn.ModuleList(convs)

                # Infer per-channel embedding dim after shared conv/pool stack.
                with torch.no_grad():
                    dummy = torch.zeros(1, 1, n_freqs, n_times)
                    dummy = self._forward_trunk(dummy)
                    self.channel_embedding_dim = int(dummy.view(1, -1).shape[1])

                concat_dim = self.n_channels * self.channel_embedding_dim

                # FC head after channel concatenation.
                fc_layers = []
                prev_dim = concat_dim
                for hidden_dim in hidden_dims:
                    fc_layers.append(nn.Linear(prev_dim, hidden_dim))
                    fc_layers.append(nn.ReLU())
                    fc_layers.append(nn.Dropout(dropout))
                    prev_dim = hidden_dim
                fc_layers.append(nn.Linear(prev_dim, n_classes))
                self.fc = nn.Sequential(*fc_layers)

            def _forward_trunk(self, x):
                for conv in self.convs:
                    x = self.relu(conv(x))
                    x = self.pool(x)
                x = self.global_pool(x)
                x = self.dropout(x)
                return x

            def forward(self, x):
                if x.ndim != 4:
                    raise ValueError(
                        "cnn_channel_concat expects 4D input "
                        "(batch, channels, freqs, times), "
                        f"got shape {tuple(x.shape)}"
                    )
                batch_size, n_channels, n_freqs, n_times = x.shape
                if n_channels != self.n_channels:
                    raise ValueError(
                        f"Channel count mismatch. Built with {self.n_channels} channels, "
                        f"got {n_channels}."
                    )

                # Per-channel shared trunk.
                channel_batch = self.channel_batch_size
                if n_channels <= channel_batch:
                    x = x.reshape(batch_size * n_channels, 1, n_freqs, n_times)
                    x = self._forward_trunk(x)
                    x = x.view(batch_size, n_channels, -1)
                else:
                    embeddings = []
                    for start in range(0, n_channels, channel_batch):
                        end = min(start + channel_batch, n_channels)
                        chunk = x[:, start:end]
                        chunk = chunk.reshape(
                            batch_size * (end - start), 1, n_freqs, n_times
                        )
                        chunk = self._forward_trunk(chunk)
                        chunk = chunk.view(batch_size, end - start, -1)
                        embeddings.append(chunk)
                    x = torch.cat(embeddings, dim=1)

                # Late channel concatenation then classification.
                x = x.reshape(batch_size, -1)
                return self.fc(x)

        return ChannelConcatCNN(
            input_shape=input_shape,
            n_classes=n_classes,
            hidden_dims=hidden_dims,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            pool_kernel=pool_kernel,
            dropout=dropout,
            channel_batch_size=channel_batch_size,
        )

    def build_model(self, input_shape, n_classes, device=None):
        """Build the model with given input shape and number of classes."""
        self._resolve_device(device)
        hidden_dims = coerce_int_list(
            self.cfg.get("hidden_dims", [128]), [128], allow_empty=True
        )
        conv_channels = coerce_int_list(
            self.cfg.get("conv_channels", [32, 64, 128]), [32, 64, 128]
        )
        kernel_size = int(self.cfg.get("kernel_size", 3))
        pool_kernel = int(self.cfg.get("pool_kernel", 2))
        dropout = float(self.cfg.get("dropout", 0.5))
        channel_batch_size = int(self.cfg.get("channel_batch_size", 0))

        self.hidden_dims = hidden_dims
        self.conv_channels = conv_channels
        self.model = self._create_network(
            input_shape=input_shape,
            n_classes=n_classes,
            hidden_dims=hidden_dims,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            pool_kernel=pool_kernel,
            dropout=dropout,
            channel_batch_size=channel_batch_size,
        )
        self.model = self.model.to(self.device)
        self.classes_ = np.arange(n_classes)
        return self.model
