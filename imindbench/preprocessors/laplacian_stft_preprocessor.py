"""
Laplacian rereferencing building blocks and Laplacian+STFT chain.
"""

import numpy as np
import torch

from . import CompositePreprocessor, register_preprocessor
from .base_preprocessor import BasePreprocessor
from .preprocessor_utils import (
    project_channel_metadata,
    validate_named_channel_sample,
)
from .stft_preprocessor import STFTPreprocessor


def laplacian_rereference_neural_data(
    electrode_data, electrode_labels, remove_non_laplacian=False
):
    """Apply Laplacian rereferencing and report surviving channel alignment.

    Returns the rereferenced data, the surviving electrode labels, and the
    original electrode indices so callers can project per-channel metadata onto
    the transformed output.
    """

    def get_all_laplacian_electrodes(electrode_labels):
        """Get all laplacian electrodes for a given subject."""

        def stem_electrode_name(name):
            found_stem_end = False
            stem, num = [], []
            for c in reversed(name):
                if c.isalpha():
                    found_stem_end = True
                if found_stem_end:
                    stem.append(c)
                else:
                    num.append(c)
            return "".join(reversed(stem)), int("".join(reversed(num)))

        def has_neighbors(stem, stems):
            (x, y) = stem
            return ((x, y + 1) in stems) or ((x, y - 1) in stems)

        def get_neighbors(stem, stems):
            (x, y) = stem
            return [f"{x}{y}" for (x, y) in [(x, y + 1), (x, y - 1)] if (x, y) in stems]

        stems = [stem_electrode_name(e) for e in electrode_labels]
        laplacian_stems = [x for x in stems if has_neighbors(x, stems)]
        electrodes = [f"{x}{y}" for (x, y) in laplacian_stems]
        neighbors = {
            e: get_neighbors(stem_electrode_name(e), stems) for e in electrodes
        }
        return electrodes, neighbors

    # Handle both numpy arrays and torch tensors
    was_tensor = isinstance(electrode_data, torch.Tensor)

    if len(electrode_data.shape) == 2:
        if was_tensor:
            electrode_data = electrode_data.unsqueeze(0)
        else:
            electrode_data = electrode_data[np.newaxis, :, :]

    # Neighbor detection works on bare channel labels even when the runtime
    # labels are subject/session prefixed.
    normalized_labels = [
        str(label).split("/")[-1] if isinstance(label, str) else str(label)
        for label in electrode_labels
    ]
    laplacian_electrodes, laplacian_neighbors = get_all_laplacian_electrodes(
        normalized_labels
    )
    laplacian_neighbor_indices = {
        laplacian_electrode_label: [
            normalized_labels.index(neighbor_label) for neighbor_label in neighbors
        ]
        for laplacian_electrode_label, neighbors in laplacian_neighbors.items()
    }
    laplacian_electrode_set = set(laplacian_electrodes)

    batch_size, n_electrodes, n_samples = electrode_data.shape
    rereferenced_n_electrodes = (
        len(laplacian_electrodes) if remove_non_laplacian else n_electrodes
    )
    if was_tensor:
        rereferenced_data = torch.zeros(
            (batch_size, rereferenced_n_electrodes, n_samples),
            dtype=electrode_data.dtype,
            device=electrode_data.device,
        )
    else:
        rereferenced_data = np.zeros(
            (batch_size, rereferenced_n_electrodes, n_samples),
            dtype=electrode_data.dtype,
        )

    electrode_i = 0
    original_electrode_indices = []
    rereferenced_labels = []
    for original_electrode_index, electrode_label in enumerate(electrode_labels):
        normalized_label = normalized_labels[original_electrode_index]
        if normalized_label in laplacian_electrode_set:
            neighbor_values = electrode_data[
                :, laplacian_neighbor_indices[normalized_label]
            ]
            neighbor_mean = (
                neighbor_values.mean(dim=1)
                if was_tensor
                else neighbor_values.mean(axis=1)
            )
            rereferenced_data[:, electrode_i] = (
                electrode_data[:, original_electrode_index] - neighbor_mean
            )
            original_electrode_indices.append(original_electrode_index)
            rereferenced_labels.append(electrode_label)
            electrode_i += 1
        else:
            if remove_non_laplacian:
                continue  # just skip the non-laplacian electrodes
            else:
                rereferenced_data[:, electrode_i] = electrode_data[
                    :, original_electrode_index
                ]
                original_electrode_indices.append(original_electrode_index)
                rereferenced_labels.append(electrode_label)
                electrode_i += 1

    return (
        rereferenced_data,
        rereferenced_labels,
        original_electrode_indices,
    )


@register_preprocessor("laplacian_rereference")
class LaplacianRereferencePreprocessor(BasePreprocessor):
    """Preprocessor that applies Laplacian rereferencing only."""

    invalidates_raw_channel_indices = True

    def _transform_one(self, sample):
        """Apply Laplacian rereferencing keyed by channel_names."""
        _, channel_ids_in, channel_names_in = validate_named_channel_sample(
            sample,
            context="laplacian_rereference",
        )

        remove_non_laplacian = self.cfg.get("remove_non_laplacian", False)
        x_out, channel_names_out, original_indices = laplacian_rereference_neural_data(
            sample["x"],
            channel_names_in,
            remove_non_laplacian=remove_non_laplacian,
        )
        if isinstance(x_out, torch.Tensor):
            x_out_np = x_out.detach().cpu().numpy()
        else:
            x_out_np = np.asarray(x_out)
        if x_out_np.ndim >= 3 and x_out_np.shape[0] == 1:
            x_out_np = x_out_np[0]

        out = project_channel_metadata(
            sample,
            context="laplacian_rereference",
            channel_ids_in=channel_ids_in,
            channel_names_in=channel_names_in,
            aligned_old_indices=original_indices,
            x_out=x_out_np,
            channel_names_out=channel_names_out,
        )
        out.pop("channel_indices", None)
        return out

    def transform_samples(self, samples):
        """Apply Laplacian rereferencing to every sample in the iterable."""
        return [self._transform_one(sample) for sample in samples]


@register_preprocessor("laplacian_stft")
class LaplacianSTFTPreprocessor(CompositePreprocessor):
    """Backwards-compatible direct class alias for the Laplacian+STFT chain."""

    invalidates_raw_channel_indices = True

    def __init__(self, cfg):
        laplacian = LaplacianRereferencePreprocessor(cfg)
        stft = STFTPreprocessor(cfg)
        super().__init__(cfg, [laplacian, stft])

    def get_context_alignment_samples(self) -> int:
        """Return the inner STFT stride needed by upstream context_window."""
        return self.preprocessors[-1].get_context_alignment_samples()
