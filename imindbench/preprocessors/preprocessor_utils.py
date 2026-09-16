"""Shared helpers for sample-local channel metadata transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def validate_named_channel_sample(
    sample: dict[str, Any],
    *,
    context: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Validate the common sample contract used by channel-local preprocessors."""
    if not isinstance(sample, dict):
        raise TypeError(f"Expected sample dict, got {type(sample).__name__}.")
    if "x" not in sample or "channel_names" not in sample:
        raise KeyError(f"{context} requires sample['x'] and sample['channel_names'].")
    if "channel_ids" not in sample:
        raise KeyError(f"{context} requires sample['channel_ids'].")

    channel_names_in = sample["channel_names"]
    if channel_names_in is None:
        raise ValueError(f"{context} requires sample['channel_names'] to be set.")
    if not isinstance(channel_names_in, list):
        raise TypeError("sample['channel_names'] must be a list[str].")
    if any(not isinstance(name, str) for name in channel_names_in):
        raise TypeError("sample['channel_names'] must contain only str values.")

    channel_ids_in = list(sample["channel_ids"])
    if len(channel_ids_in) != len(channel_names_in):
        raise ValueError(
            "sample['channel_ids'] and sample['channel_names'] length mismatch: "
            f"{len(channel_ids_in)} vs {len(channel_names_in)}."
        )

    x_in = np.asarray(sample["x"])
    if x_in.ndim < 2:
        raise ValueError(
            f"{context} expects sample['x'] to be at least 2D "
            f"(channels, features...), got {x_in.shape}."
        )
    if x_in.shape[0] != len(channel_names_in):
        raise ValueError(
            "sample['x'] channel axis must match channel_names length: "
            f"{x_in.shape[0]} vs {len(channel_names_in)}."
        )

    return x_in, channel_ids_in, channel_names_in


def project_channel_metadata(
    sample: dict[str, Any],
    *,
    context: str,
    channel_ids_in: list[str],
    channel_names_in: list[str],
    aligned_old_indices: Sequence[int],
    x_out: Any,
    channel_names_out: Sequence[str],
) -> dict[str, Any]:
    """Return one sample with channel-local metadata projected to kept indices."""
    aligned_old_indices = [int(idx) for idx in aligned_old_indices]
    channel_names_out = list(channel_names_out)
    x_out_np = np.asarray(x_out, dtype=np.float32)

    if x_out_np.ndim < 2:
        raise ValueError(
            f"{context} output must be at least 2D (channels, features...), "
            f"got shape {x_out_np.shape}."
        )
    if x_out_np.shape[0] != len(channel_names_out):
        raise ValueError(
            f"{context} output channel axis does not match returned labels: "
            f"{x_out_np.shape[0]} vs {len(channel_names_out)}."
        )
    if len(aligned_old_indices) != len(channel_names_out):
        raise ValueError(
            f"{context} returned mismatched metadata indices: "
            f"{len(aligned_old_indices)} vs {len(channel_names_out)}."
        )

    expected_len = len(channel_names_in)
    out = dict(sample)
    out["x"] = x_out_np
    out["channel_names"] = channel_names_out
    # channel_ids / coords / seq_id / brain_areas all stay aligned to the
    # original channel axis, so every channel-local transform can project them
    # with the same retained-index list.
    out["channel_ids"] = [channel_ids_in[i] for i in aligned_old_indices]

    coords = sample.get("channel_coords")
    if coords is not None:
        coords_arr = np.asarray(coords)
        if coords_arr.shape[0] != expected_len:
            raise ValueError(
                "channel_coords first dimension mismatch: "
                f"{coords_arr.shape[0]} vs {expected_len}."
            )
        out["channel_coords"] = coords_arr[aligned_old_indices]

    seq_id = sample.get("seq_id")
    if seq_id is not None:
        seq_arr = np.asarray(seq_id)
        if seq_arr.shape[0] != expected_len:
            raise ValueError(
                f"seq_id first dimension mismatch: {seq_arr.shape[0]} vs {expected_len}."
            )
        out["seq_id"] = seq_arr[aligned_old_indices]

    brain_areas = sample.get("brain_areas")
    if brain_areas is not None:
        brain_areas_arr = np.asarray(brain_areas, dtype=object).reshape(-1)
        if brain_areas_arr.shape[0] != expected_len:
            raise ValueError(
                "brain_areas first dimension mismatch: "
                f"{brain_areas_arr.shape[0]} vs {expected_len}."
            )
        out["brain_areas"] = brain_areas_arr[aligned_old_indices].tolist()

    channel_indices = sample.get("channel_indices")
    if channel_indices is not None:
        channel_indices_arr = np.asarray(channel_indices, dtype=np.int64).reshape(-1)
        if channel_indices_arr.shape[0] != expected_len:
            raise ValueError(
                "channel_indices first dimension mismatch: "
                f"{channel_indices_arr.shape[0]} vs {expected_len}."
            )
        out["channel_indices"] = channel_indices_arr[aligned_old_indices]

    return out
