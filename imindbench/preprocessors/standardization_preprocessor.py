"""
Standardization preprocessor for variable-channel pipelines.

Canonical sample axis convention is ``(channels, time, features...)``. For
STFT samples this means ``(channels, timebins, freqs)``.
"""

from __future__ import annotations

from copy import deepcopy
import numpy as np
from .base_preprocessor import BasePreprocessor
from . import register_preprocessor


@register_preprocessor("standardize")
class StandardizationPreprocessor(BasePreprocessor):
    """
    Preprocessor that standardizes sample dictionaries with configurable modes.

    Modes:
        - 'global_feature': one mean/std per feature/time index, pooled over
          train samples and channels. This preserves the full axis-1+ feature
          grid, so STFT timebins and freqs each keep separate stats.
        - 'global_scalar': one mean/std scalar pooled over every train value,
          flattening channels, axis-1 time/timebins, and all feature axes.
          Intended mainly for waveform inputs; for STFT this mixes frequency
          bins with different natural scales.
        - 'global_robust_scalar': one approximate median/MAD scale scalar
          pooled over a bounded reservoir of flattened train values. Intended
          mainly for waveform inputs; for STFT/multi-STFT use as an ablation
          rather than a default.
        - 'per_channel_feature': feature-wise mean/std tracked per channel_id,
          with optional unseen-channel fallback policy. Like global_feature,
          each axis-1+ feature/time index keeps separate stats.
        - 'per_channel_samples_time_pooled': per-channel-id stats pooled across
          sample and axis-1 time/timebin observations. Remaining axes are
          features, so STFT frequency bins are the feature axis.
        - 'sample_per_channel_time': per-sample, per-channel stats over the
          axis-1 time/timebin axis. For STFT-like x, frequency/features are
          kept separate.

    Recommended use:
        - For waveform inputs shaped (channels, time), global_scalar and
          global_robust_scalar can normalize recording scale while preserving
          cross-channel amplitude relationships.
        - For STFT/multi-STFT inputs shaped (channels, timebins, freqs),
          prefer frequency-aware modes such as global_feature or
          per_channel_samples_time_pooled. A single global scalar is usually
          only appropriate as an ablation because frequency bins can have very
          different natural scales.
    """

    is_standardization_stage = True

    VARIABLE_MODES = {
        "global_feature",
        "global_scalar",
        "global_robust_scalar",
        "per_channel_feature",
        "per_channel_samples_time_pooled",
    }
    SAMPLE_LOCAL_MODES = {
        "sample_per_channel_time",
    }
    CHANNEL_ID_VARIABLE_MODES = {
        "per_channel_feature",
        "per_channel_samples_time_pooled",
    }
    VALID_MODES = VARIABLE_MODES | SAMPLE_LOCAL_MODES

    def __init__(self, cfg):
        super().__init__(cfg)
        self.mode = getattr(cfg, "mode", "per_channel_feature")
        self._state = None
        self.eps = float(getattr(cfg, "eps", 1e-8))
        self.unseen_channel_policy = str(
            getattr(cfg, "unseen_channel_policy", "global_fallback")
        )
        self.robust_reservoir_size = int(
            getattr(cfg, "robust_reservoir_size", 1_000_000)
        )
        self.robust_random_seed = int(getattr(cfg, "robust_random_seed", 0))

        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of {sorted(self.VALID_MODES)}"
            )

        if self.mode == "global_robust_scalar" and self.robust_reservoir_size <= 0:
            raise ValueError("robust_reservoir_size must be a positive integer.")

        if self.mode in self.CHANNEL_ID_VARIABLE_MODES:
            valid_policies = {"global_fallback", "error"}
            if self.unseen_channel_policy not in valid_policies:
                raise ValueError(
                    "Invalid unseen_channel_policy "
                    f"'{self.unseen_channel_policy}'. Expected one of {sorted(valid_policies)}."
                )

        self.execution_type = (
            "sample_local"
            if self.mode in self.SAMPLE_LOCAL_MODES
            else "fold_fit_transform"
        )

    def fit_split(self, sample_iter):
        state = self._new_variable_state()
        saw_sample = False
        for sample in sample_iter:
            saw_sample = True
            if self.mode == "global_feature":
                x_matrix, feature_shape = self._extract_feature_matrix(sample)
                self._ensure_feature_shape(state, feature_shape)
                self._update_stats_matrix(state["global"], x_matrix)
                continue

            if self.mode == "global_scalar":
                x_matrix = self._extract_global_scalar_matrix(sample)
                self._update_stats_matrix(state["global"], x_matrix)
                continue

            if self.mode == "global_robust_scalar":
                x_matrix = self._extract_global_scalar_matrix(sample)
                self._update_robust_reservoir(state["global"], x_matrix)
                continue

            if self.mode == "per_channel_feature":
                x_matrix, feature_shape = self._extract_feature_matrix(sample)
                self._ensure_feature_shape(state, feature_shape)
                channel_ids = self._require_channel_ids(
                    sample, expected_n=x_matrix.shape[0]
                )
                for channel_id, row in zip(channel_ids, x_matrix):
                    ch_key = str(channel_id)
                    ch_stats = state["per_channel"].setdefault(
                        ch_key,
                        self._new_stats_accumulator(),
                    )
                    self._update_stats_row(ch_stats, row)
                    self._update_stats_row(state["global"], row)
                continue

            if self.mode == "per_channel_samples_time_pooled":
                x_matrix, feature_shape = self._extract_channel_time_feature_matrix(
                    sample
                )
                self._ensure_feature_shape(state, feature_shape)
                channel_ids = self._require_channel_ids(
                    sample, expected_n=x_matrix.shape[0]
                )
                for channel_id, channel_matrix in zip(channel_ids, x_matrix):
                    ch_key = str(channel_id)
                    ch_stats = state["per_channel"].setdefault(
                        ch_key,
                        self._new_stats_accumulator(),
                    )
                    self._update_stats_matrix(ch_stats, channel_matrix)
                    self._update_stats_matrix(state["global"], channel_matrix)
                continue

            raise ValueError(f"Unsupported variable mode: {self.mode}")

        if not saw_sample:
            raise ValueError("fit_split received zero samples.")

        if state["global"]["count"] == 0:
            raise ValueError("fit_split produced empty statistics.")

        if self.mode == "global_robust_scalar":
            self._finalize_global_robust_state(state)

        self._state = state
        return self.get_state()

    def set_state(self, state):
        if state is None:
            self._state = None
            return
        decoded = self._decode_state(state)
        # Loaded state is authoritative for transform behavior.
        self._state = decoded
        self.eps = float(decoded["eps"])
        self.unseen_channel_policy = str(decoded["unseen_channel_policy"])
        self.robust_reservoir_size = int(decoded["robust_reservoir_size"])
        self.robust_random_seed = int(decoded["robust_random_seed"])

    def get_state(self):
        if self._state is None:
            return None
        return self._encode_state(self._state)

    def _transform_one(self, sample):
        if self.mode == "sample_per_channel_time":
            if "x" not in sample:
                raise KeyError("sample dict must contain key 'x'.")
            x = np.asarray(sample["x"], dtype=np.float64)
            if x.ndim < 2:
                raise ValueError(
                    "sample['x'] must be at least 2D (channels, time, ...), "
                    f"got {x.shape}."
                )
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            if self.eps > 0.0:
                std = np.maximum(std, float(self.eps))
            std[std == 0] = 1.0
            out = dict(sample)
            out["x"] = ((x - mean) / std).astype(np.float32, copy=False)
            return out

        if self._state is None:
            raise RuntimeError(
                "Variable-channel standardization state is unset. "
                "Call fit_split(...) then set_state(...)/use fitted instance."
            )

        if self.mode == "global_feature":
            x_matrix, feature_shape = self._extract_feature_matrix(sample)
            self._ensure_feature_shape(self._state, feature_shape)
            mean, std = self._stats_mean_std(self._state["global"])
            normalized = (x_matrix - mean[None, :]) / std[None, :]
            out = dict(sample)
            out["x"] = normalized.reshape(sample["x"].shape).astype(np.float32)
            return out

        if self.mode == "global_scalar":
            x = np.asarray(sample["x"], dtype=np.float64)
            mean, std = self._stats_mean_std(self._state["global"])
            normalized = (x - float(mean[0])) / float(std[0])
            out = dict(sample)
            out["x"] = normalized.astype(np.float32)
            return out

        if self.mode == "global_robust_scalar":
            x = np.asarray(sample["x"], dtype=np.float64)
            center = np.asarray(self._state["global"]["center"], dtype=np.float64)
            scale = np.asarray(self._state["global"]["scale"], dtype=np.float64)
            denom = max(float(scale[0]), self.eps)
            if denom == 0.0:
                denom = 1.0
            normalized = (x - float(center[0])) / denom
            out = dict(sample)
            out["x"] = normalized.astype(np.float32)
            return out

        if self.mode == "per_channel_feature":
            x_matrix, feature_shape = self._extract_feature_matrix(sample)
            self._ensure_feature_shape(self._state, feature_shape)
            channel_ids = self._require_channel_ids(
                sample, expected_n=x_matrix.shape[0]
            )
            normalized = np.empty_like(x_matrix, dtype=np.float32)
            global_mean, global_std = self._stats_mean_std(self._state["global"])
            for i, channel_id in enumerate(channel_ids):
                ch_stats = self._state["per_channel"].get(str(channel_id))
                if ch_stats is None:
                    if self.unseen_channel_policy == "error":
                        raise KeyError(
                            "Unseen channel_id during eval: "
                            f"'{channel_id}'. Set unseen_channel_policy=global_fallback to allow fallback."
                        )
                    mean, std = global_mean, global_std
                else:
                    mean, std = self._stats_mean_std(ch_stats)
                normalized[i] = (x_matrix[i] - mean) / std
            out = dict(sample)
            out["x"] = normalized.reshape(sample["x"].shape).astype(np.float32)
            return out

        if self.mode == "per_channel_samples_time_pooled":
            x_matrix, feature_shape = self._extract_channel_time_feature_matrix(sample)
            self._ensure_feature_shape(self._state, feature_shape)
            channel_ids = self._require_channel_ids(
                sample, expected_n=x_matrix.shape[0]
            )
            normalized = np.empty_like(x_matrix, dtype=np.float32)
            global_mean, global_std = self._stats_mean_std(self._state["global"])
            for i, channel_id in enumerate(channel_ids):
                ch_stats = self._state["per_channel"].get(str(channel_id))
                if ch_stats is None:
                    if self.unseen_channel_policy == "error":
                        raise KeyError(
                            "Unseen channel_id during eval: "
                            f"'{channel_id}'. Set unseen_channel_policy=global_fallback to allow fallback."
                        )
                    mean, std = global_mean, global_std
                else:
                    mean, std = self._stats_mean_std(ch_stats)
                normalized[i] = (x_matrix[i] - mean[None, :]) / std[None, :]
            out = dict(sample)
            out["x"] = normalized.reshape(sample["x"].shape).astype(np.float32)
            return out

        raise ValueError(f"Unsupported variable mode: {self.mode}")

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]

    def reset_state(self):
        """Reset fitted variable-channel state."""
        self._state = None

    @staticmethod
    def _new_stats_accumulator():
        return {"count": 0, "mean": None, "m2": None}

    def _new_robust_stats_accumulator(self):
        return {
            "count": 0,
            "reservoir": np.empty((self.robust_reservoir_size, 1), dtype=np.float64),
            "reservoir_size": self.robust_reservoir_size,
            "rng": np.random.default_rng(self.robust_random_seed),
            "sample_count": 0,
            "center": None,
            "scale": None,
        }

    def _new_variable_state(self):
        return {
            "mode": self.mode,
            "feature_shape": (
                ("scalar",)
                if self.mode in {"global_scalar", "global_robust_scalar"}
                else None
            ),
            "global": (
                self._new_robust_stats_accumulator()
                if self.mode == "global_robust_scalar"
                else self._new_stats_accumulator()
            ),
            "per_channel": {},
            "eps": self.eps,
            "unseen_channel_policy": self.unseen_channel_policy,
        }

    @staticmethod
    def _extract_feature_matrix(sample):
        if "x" not in sample:
            raise KeyError("sample dict must contain key 'x'.")
        x = np.asarray(sample["x"])
        if x.ndim < 2:
            raise ValueError(
                f"sample['x'] must be at least 2D (channels, features...), got {x.shape}."
            )
        channels = x.shape[0]
        feature_shape = tuple(x.shape[1:])
        return x.reshape(channels, -1), feature_shape

    @staticmethod
    def _extract_channel_time_feature_matrix(sample):
        if "x" not in sample:
            raise KeyError("sample dict must contain key 'x'.")
        x = np.asarray(sample["x"])
        if x.ndim < 2:
            raise ValueError(
                "sample['x'] must be at least 2D (channels, time, ...), "
                f"got {x.shape}."
            )
        channels = int(x.shape[0])
        time_steps = int(x.shape[1])
        pooled_feature_shape = tuple(x.shape[2:]) or (1,)
        return x.reshape(channels, time_steps, -1), pooled_feature_shape

    @staticmethod
    def _extract_global_scalar_matrix(sample):
        if "x" not in sample:
            raise KeyError("sample dict must contain key 'x'.")
        x = np.asarray(sample["x"])
        if x.ndim < 2:
            raise ValueError(
                "sample['x'] must be at least 2D (channels, time, ...), "
                f"got {x.shape}."
            )
        return x.reshape(-1, 1)

    @staticmethod
    def _require_channel_ids(sample, *, expected_n):
        channel_ids = sample.get("channel_ids")
        if channel_ids is None:
            raise KeyError("Channel-aware mode requires sample['channel_ids'].")
        if len(channel_ids) != expected_n:
            raise ValueError(
                "channel_ids length must match x channels: "
                f"{len(channel_ids)} vs {expected_n}."
            )
        return channel_ids

    @staticmethod
    def _ensure_feature_shape(state, feature_shape):
        if state["feature_shape"] is None:
            state["feature_shape"] = tuple(feature_shape)
            return
        if tuple(state["feature_shape"]) != tuple(feature_shape):
            raise ValueError(
                "Feature shape mismatch across samples: "
                f"expected {tuple(state['feature_shape'])}, got {tuple(feature_shape)}."
            )

    def _update_stats_matrix(self, acc, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError(
                f"Expected 2D matrix for stats update, got shape {matrix.shape}."
            )
        n_rows = int(matrix.shape[0])
        if n_rows == 0:
            return

        # Aggregate the incoming batch once, then merge it into the running
        # Welford state. This keeps pooled standardization numerically stable
        # while avoiding the Python row loop that dominated fit_split time.
        batch_mean = matrix.mean(axis=0, dtype=np.float64)
        centered = matrix - batch_mean[None, :]
        batch_m2 = (centered * centered).sum(axis=0, dtype=np.float64)

        if acc["mean"] is None:
            acc["count"] = n_rows
            acc["mean"] = batch_mean
            acc["m2"] = batch_m2
            return

        prev_count = int(acc["count"])
        total_count = prev_count + n_rows
        delta = batch_mean - acc["mean"]
        # Parallel-variance merge: update the existing accumulator with the
        # batch summary without materializing per-row intermediate states.
        acc["mean"] += delta * (float(n_rows) / float(total_count))
        acc["m2"] += batch_m2 + (delta * delta) * (
            float(prev_count) * float(n_rows) / float(total_count)
        )
        acc["count"] = total_count

    @staticmethod
    def _update_stats_row(acc, row):
        row = np.asarray(row, dtype=np.float64)
        if acc["mean"] is None:
            acc["mean"] = np.zeros_like(row, dtype=np.float64)
            acc["m2"] = np.zeros_like(row, dtype=np.float64)
        acc["count"] += 1
        delta = row - acc["mean"]
        acc["mean"] += delta / acc["count"]
        delta2 = row - acc["mean"]
        acc["m2"] += delta * delta2

    def _stats_mean_std(self, acc):
        count = int(acc["count"])
        mean = np.asarray(acc["mean"], dtype=np.float64)
        if count <= 1:
            std = np.ones_like(mean, dtype=np.float64)
            return mean.astype(np.float32), std.astype(np.float32)
        m2 = np.asarray(acc["m2"], dtype=np.float64)
        # Match sklearn StandardScaler variance convention (population variance, ddof=0).
        var = m2 / float(max(count, 1))
        std = np.sqrt(np.maximum(var, 0.0))
        if self.eps > 0.0:
            std = np.maximum(std, float(self.eps))
        std[std == 0] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    @staticmethod
    def _update_robust_reservoir(acc, matrix):
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(
                f"Expected 2D matrix for robust update, got shape {values.shape}."
            )
        if values.shape[0] == 0:
            return

        reservoir_size = int(acc["reservoir_size"])
        reservoir = acc["reservoir"]
        rng = acc["rng"]
        seen = int(acc["count"])
        sample_count = int(acc["sample_count"])

        fill_count = min(reservoir_size - sample_count, values.shape[0])
        if fill_count > 0:
            reservoir[sample_count : sample_count + fill_count] = values[:fill_count]
            sample_count += fill_count
            seen += fill_count

        for row in values[fill_count:]:
            seen += 1
            replace_idx = int(rng.integers(0, seen))
            if replace_idx < reservoir_size:
                reservoir[replace_idx] = row

        acc["count"] = seen
        acc["reservoir"] = reservoir
        acc["sample_count"] = sample_count

    @staticmethod
    def _finalize_global_robust_state(state):
        reservoir = state["global"].pop("reservoir", None)
        state["global"].pop("rng", None)
        sample_count = int(state["global"].get("sample_count", 0))
        if reservoir is None or sample_count == 0:
            raise ValueError("global_robust_scalar produced empty statistics.")
        pooled = np.asarray(reservoir[:sample_count], dtype=np.float64)
        center = np.median(pooled, axis=0)
        mad = np.median(np.abs(pooled - center[None, :]), axis=0)
        scale = 1.4826 * mad
        eps = float(state["eps"])
        if eps > 0.0:
            scale = np.maximum(scale, eps)
        scale[scale == 0] = 1.0
        state["global"]["center"] = center
        state["global"]["scale"] = scale

    @staticmethod
    def _encode_acc(acc):
        mean = acc["mean"]
        m2 = acc["m2"]
        if mean is None:
            return {"count": int(acc["count"]), "mean": [], "m2": []}
        return {
            "count": int(acc["count"]),
            "mean": np.asarray(mean, dtype=np.float32).tolist(),
            "m2": np.asarray(m2, dtype=np.float32).tolist(),
        }

    @staticmethod
    def _encode_robust_acc(acc):
        center = acc["center"]
        scale = acc["scale"]
        if center is None or scale is None:
            return {"count": int(acc["count"]), "center": [], "scale": []}
        return {
            "count": int(acc["count"]),
            "sample_count": int(acc.get("sample_count", acc["count"])),
            "reservoir_size": int(acc.get("reservoir_size", acc["count"])),
            "center": np.asarray(center, dtype=np.float32).tolist(),
            "scale": np.asarray(scale, dtype=np.float32).tolist(),
        }

    @staticmethod
    def _decode_acc(encoded):
        count = int(encoded.get("count", 0))
        mean_list = encoded.get("mean", [])
        m2_list = encoded.get("m2", [])
        if count <= 0 or len(mean_list) == 0:
            return {"count": 0, "mean": None, "m2": None}
        mean = np.asarray(mean_list, dtype=np.float64)
        m2 = np.asarray(m2_list, dtype=np.float64)
        if mean.shape != m2.shape:
            raise ValueError("Encoded stats mean/m2 shape mismatch.")
        return {"count": count, "mean": mean, "m2": m2}

    @staticmethod
    def _decode_robust_acc(encoded):
        count = int(encoded.get("count", 0))
        center_list = encoded.get("center", [])
        scale_list = encoded.get("scale", [])
        if count <= 0 or len(center_list) == 0:
            return {
                "count": 0,
                "reservoir": np.empty((0, 1), dtype=np.float64),
                "reservoir_size": int(encoded.get("reservoir_size", 0)),
                "sample_count": int(encoded.get("sample_count", 0)),
                "center": None,
                "scale": None,
            }
        center = np.asarray(center_list, dtype=np.float64)
        scale = np.asarray(scale_list, dtype=np.float64)
        if center.shape != scale.shape:
            raise ValueError("Encoded robust stats center/scale shape mismatch.")
        return {
            "count": count,
            "reservoir": np.empty((0, center.size), dtype=np.float64),
            "reservoir_size": int(encoded.get("reservoir_size", count)),
            "sample_count": int(encoded.get("sample_count", count)),
            "center": center,
            "scale": scale,
        }

    def _encode_state(self, state):
        return {
            "mode": state["mode"],
            "feature_shape": list(state["feature_shape"]),
            "global": (
                self._encode_robust_acc(state["global"])
                if state["mode"] == "global_robust_scalar"
                else self._encode_acc(state["global"])
            ),
            "per_channel": {
                key: self._encode_acc(acc) for key, acc in state["per_channel"].items()
            },
            "eps": float(state["eps"]),
            "unseen_channel_policy": state["unseen_channel_policy"],
            "robust_reservoir_size": int(self.robust_reservoir_size),
            "robust_random_seed": int(self.robust_random_seed),
        }

    def _decode_state(self, state):
        mode = state.get("mode", self.mode)
        if mode != self.mode:
            raise ValueError(
                "Standardization state mode mismatch: "
                f"state='{mode}', instance='{self.mode}'."
            )
        feature_shape = tuple(state.get("feature_shape", ()))
        if not feature_shape:
            raise ValueError("Standardization state missing feature_shape.")
        decoded = {
            "mode": mode,
            "feature_shape": feature_shape,
            "global": (
                self._decode_robust_acc(state["global"])
                if mode == "global_robust_scalar"
                else self._decode_acc(state["global"])
            ),
            "per_channel": {
                str(key): self._decode_acc(acc)
                for key, acc in state.get("per_channel", {}).items()
            },
            "eps": float(state.get("eps", self.eps)),
            "unseen_channel_policy": str(
                state.get("unseen_channel_policy", self.unseen_channel_policy)
            ),
            "robust_reservoir_size": int(
                state.get("robust_reservoir_size", self.robust_reservoir_size)
            ),
            "robust_random_seed": int(
                state.get("robust_random_seed", self.robust_random_seed)
            ),
        }
        return deepcopy(decoded)
