"""Explicit waveform window policies without mutating TorchBrain's time-series API."""

import math

import numpy as np

DEFAULT_WINDOW_SLICING_POLICY = "ceil"


def validate_window_slicing_policy(policy) -> None:
    if not isinstance(policy, str) or policy not in {"ceil", "legacy_floor"}:
        raise ValueError(
            "dataset.window_slicing_policy must be 'ceil' or 'legacy_floor', "
            f"got {policy!r}."
        )


def read_recording_window(
    recording, start: float, end: float, policy: str
) -> np.ndarray:
    """Read only the requested waveform interval, retaining lazy storage slicing."""
    if policy == "ceil":
        # Preserve the current provider/TorchBrain slicing path exactly.
        return np.asarray(recording.slice(start, end).seeg_data.data)
    if policy != "legacy_floor":
        raise ValueError(f"Unknown window slicing policy: {policy!r}.")

    series = recording.seeg_data
    origin = float(series.domain.start[0])
    domain_end = float(series.domain.end[-1])
    rate = float(series.sampling_rate)

    def floor_boundary(time):
        if time <= origin:
            return origin
        if time > domain_end:
            return domain_end
        index = math.floor((time - origin) * rate)
        return origin + index / rate

    slice_start, slice_end = floor_boundary(start), floor_boundary(end)
    # Snap reconstructed grid points despite cancellation at large time origins.
    # Never apply this tolerance to the original unsnapped historical floor rule.
    eps = max(
        1e-9,
        4 * rate * max(abs(np.spacing(t)) for t in (origin, slice_start, slice_end)),
    )
    window = series.slice(slice_start, slice_end, eps=eps)
    return np.asarray(window.data)
