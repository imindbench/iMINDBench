"""
BrainBERT encoder preprocessor for extracting temporal embeddings.
"""

from collections.abc import Iterable

import torch
import numpy as np
from .base_preprocessor import BasePreprocessor
from . import register_preprocessor
from neuroprobe_eval.utils.logging_utils import log


@register_preprocessor("brainbert_encoder")
class BrainBERTEncoderPreprocessor(BasePreprocessor):
    """BrainBERT encoder preprocessor for extracting temporal embeddings."""

    execution_type = "fold_fit_transform"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.upstream_ckpt = cfg.upstream_ckpt
        self.device = cfg.get("device", "cpu")
        self.clip_emb_k = cfg.get("clip_emb_k", 5)
        # Pooling mode for encoder outputs: mean, max, or raw.
        self.pool = cfg.get("pool", "mean")

        # Adaptive batching config.
        self.max_memory_fraction = cfg.get("max_memory_fraction", 0.75)
        self.manual_batch_size = cfg.get("encoder_batch_size", None)
        self.min_batch_size = cfg.get("min_batch_size", 32)
        self.verbose = cfg.get("verbose", False)

        # Lazy-load the encoder so config validation/import does not require the
        # checkpoint to be present.
        self.model = None
        self.model_cfg = None
        self._optimal_batch_size = None
        self._state = None

    def _is_cuda_device(self):
        """Whether the configured encoder device is CUDA-backed."""
        return str(self.device).startswith("cuda")

    def _load_model(self):
        """Load the BrainBERT encoder checkpoint on first use."""
        if self.model is None:
            from neuroprobe_eval.models.popt_components.brainbert_encoder import (
                BrainBERTEncoder,
            )

            self.model, self.model_cfg = BrainBERTEncoder.from_checkpoint(
                self.upstream_ckpt, device=self.device
            )
            self.model.to(self.device)
            self.model.eval()

    def unload_model(self):
        """Unload the encoder from GPU memory after preprocessing completes."""
        if self.model is not None and self._is_cuda_device():
            self.model.cpu()
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            log("[BrainBERT] Model unloaded from GPU to free memory", priority=1)

    def _estimate_optimal_batch_size(self, n_time, n_freq):
        """Estimate a safe encoder batch size from sequence shape and GPU memory."""
        # Manual override takes precedence
        if self.manual_batch_size is not None:
            return self.manual_batch_size

        # No limit on CPU
        if not self._is_cuda_device():
            return 999999

        # Query GPU memory
        gpu_props = torch.cuda.get_device_properties(self.device)
        total_memory = gpu_props.total_memory
        reserved = torch.cuda.memory_reserved(self.device)
        available = total_memory - reserved

        # Get model architecture params (with fallbacks for safety)
        if self.model_cfg:
            hidden_dim = getattr(self.model_cfg, "hidden_dim", 768)
            nhead = getattr(self.model_cfg, "nhead", 12)
            num_layers = getattr(self.model_cfg, "encoder_num_layers", 12)
            dim_feedforward = getattr(self.model_cfg, "layer_dim_feedforward", 3072)
        else:
            # Fallback defaults (shouldn't happen if model is loaded)
            hidden_dim = 768
            nhead = 12
            num_layers = 12
            dim_feedforward = 3072

        # Estimate memory per sequence with accurate formula:
        # Input tensor
        input_memory = n_time * n_freq * 4

        # Per-layer memory (this is where the real cost is):
        # - Attention: nhead * seq_len^2 (quadratic in sequence length!)
        # - FFN intermediate: seq_len * dim_feedforward (large!)
        # - Layer output: seq_len * hidden_dim
        attention_memory = nhead * n_time * n_time * 4
        ffn_memory = n_time * dim_feedforward * 4
        output_memory = n_time * hidden_dim * 4
        per_layer_memory = attention_memory + ffn_memory + output_memory

        # Total across all layers + 40% overhead for temporary tensors
        bytes_per_sequence = (input_memory + per_layer_memory * num_layers) * 1.4

        # Calculate with safety margin
        usable_memory = available * self.max_memory_fraction
        optimal_batch = int(usable_memory / bytes_per_sequence)

        # Debug logging
        if self.verbose:
            log("[BrainBERT Memory Estimation]", priority=2)
            log(f"  GPU available: {usable_memory / 1e9:.2f} GB", priority=2, indent=1)
            log(
                f"  Per-sequence estimate: {bytes_per_sequence / 1e6:.2f} MB",
                priority=2,
                indent=1,
            )
            log(f"    - Input: {input_memory / 1e3:.1f} KB", priority=2, indent=2)
            log(
                f"    - Per layer: {per_layer_memory / 1e3:.1f} KB × {num_layers} layers",
                priority=2,
                indent=2,
            )
            log(
                f"    - Attention (quadratic): {attention_memory / 1e3:.1f} KB",
                priority=2,
                indent=2,
            )
            log(f"    - FFN: {ffn_memory / 1e3:.1f} KB", priority=2, indent=2)
            log(f"  Calculated batch size: {optimal_batch}", priority=2, indent=1)

        # Check if even minimum batch size fits
        min_batch_memory = bytes_per_sequence * self.min_batch_size
        if min_batch_memory > usable_memory:
            # Return negative value to signal insufficient memory
            return -1

        # Caller enforces the configured minimum batch size when execution starts.
        return optimal_batch

    def _print_memory_error(self, n_time, n_freq, total_sequences):
        """Log a detailed memory error when encoder batching cannot fit on GPU."""
        if not self._is_cuda_device():
            return

        gpu_props = torch.cuda.get_device_properties(self.device)
        total_gb = gpu_props.total_memory / 1e9
        reserved_gb = torch.cuda.memory_reserved(self.device) / 1e9
        allocated_gb = torch.cuda.memory_allocated(self.device) / 1e9

        log("=" * 60, priority=0)
        log("GPU MEMORY ERROR", priority=0)
        log("=" * 60, priority=0)
        log(f"GPU: {gpu_props.name}", priority=0)
        log(f"Total memory: {total_gb:.2f} GB", priority=0)
        log(f"Reserved: {reserved_gb:.2f} GB", priority=0)
        log(f"Allocated: {allocated_gb:.2f} GB", priority=0)
        log(
            f"Requested: {total_sequences} sequences of shape ({n_time}, {n_freq})",
            priority=0,
        )
        log("Suggestions:", priority=0)
        log("  1. Reduce number of samples processed at once", priority=0, indent=1)
        log("  2. Use smaller model or reduce hidden_dim", priority=0, indent=1)
        log("  3. Set encoder_batch_size=32 manually in config", priority=0, indent=1)
        log("  4. Process on CPU (slower): device='cpu'", priority=0, indent=1)
        log("=" * 60, priority=0)

    def _process_in_chunks(self, data_reshaped, chunk_size):
        """Process large batch in smaller chunks.

        Runs encoder + postprocessing per chunk on device, then moves reduced
        outputs to CPU for concatenation.
        """
        total_sequences = data_reshaped.shape[0]
        n_time = data_reshaped.shape[1]
        all_outputs = []

        for start_idx in range(0, total_sequences, chunk_size):
            end_idx = min(start_idx + chunk_size, total_sequences)
            chunk = data_reshaped[start_idx:end_idx].to(self.device)

            with torch.no_grad():
                chunk_output = self.model(chunk, src_key_mask=None)
            chunk_output = self._postprocess_encoder_output(chunk_output, n_time=n_time)

            # Keep only reduced chunk outputs on CPU; avoid moving the full
            # concatenated tensor back to device.
            all_outputs.append(chunk_output.cpu())

        return torch.cat(all_outputs, dim=0)

    def _postprocess_encoder_output(self, outputs, *, n_time):
        """Apply pooling/clipping to raw encoder outputs."""
        middle = n_time // 2
        if self.clip_emb_k > 0 and n_time >= self.clip_emb_k * 2:
            outputs = outputs[:, middle - self.clip_emb_k : middle + self.clip_emb_k, :]

        if self.pool == "mean":
            return outputs.mean(dim=1)
        if self.pool == "max":
            return outputs.max(dim=1)[0]
        if self.pool == "raw":
            return outputs
        raise ValueError(f"Unknown pool type: {self.pool}")

    @staticmethod
    def _extract_sample_feature_shape(sample):
        if not isinstance(sample, dict):
            raise TypeError(
                "brainbert_encoder expects sample dicts, got "
                f"{type(sample).__name__}."
            )
        if "x" not in sample:
            raise KeyError("brainbert_encoder requires sample['x'].")
        x = np.asarray(sample["x"])
        if x.ndim != 3:
            raise ValueError(
                "brainbert_encoder requires sample['x'] with shape "
                f"(channels, time, freq), got {x.shape}."
            )
        return int(x.shape[1]), int(x.shape[2])

    def fit_split(self, sample_iter: Iterable[dict]) -> dict:
        """Fit split-level encoder metadata such as feature shape and batch size."""
        self._state = None
        self._optimal_batch_size = None
        self._load_model()
        feature_shape = None
        saw_sample = False
        for sample in sample_iter:
            saw_sample = True
            sample_shape = self._extract_sample_feature_shape(sample)
            if feature_shape is None:
                feature_shape = sample_shape
            elif sample_shape != feature_shape:
                raise ValueError(
                    "brainbert_encoder requires consistent (time, freq) dimensions "
                    f"within a split, got {feature_shape} and {sample_shape}."
                )

        if not saw_sample or feature_shape is None:
            raise ValueError("brainbert_encoder fit_split received zero samples.")

        n_time, n_freq = feature_shape
        estimated_batch = int(self._estimate_optimal_batch_size(n_time, n_freq))
        if estimated_batch < 0:
            self._print_memory_error(n_time, n_freq, self.min_batch_size)
            raise RuntimeError(
                "Insufficient GPU memory for brainbert_encoder preprocessing."
            )

        self._optimal_batch_size = estimated_batch
        self._state = {
            "feature_shape": [n_time, n_freq],
            "optimal_batch_size": estimated_batch,
            "pool": str(self.pool),
            "clip_emb_k": int(self.clip_emb_k),
        }
        return self.get_state() or {}

    def set_state(self, state):
        if state is None:
            self._state = None
            self._optimal_batch_size = None
            return

        if not isinstance(state, dict):
            raise TypeError(
                "brainbert_encoder state must be a dict, got "
                f"{type(state).__name__}."
            )
        feature_shape = state.get("feature_shape")
        if not isinstance(feature_shape, (list, tuple)) or len(feature_shape) != 2:
            raise ValueError(
                "brainbert_encoder state.feature_shape must be [time, freq]."
            )
        n_time = int(feature_shape[0])
        n_freq = int(feature_shape[1])
        if n_time <= 0 or n_freq <= 0:
            raise ValueError(
                "brainbert_encoder state.feature_shape values must be positive."
            )

        optimal_batch_size = int(
            state.get(
                "optimal_batch_size",
                self.manual_batch_size if self.manual_batch_size is not None else 1,
            )
        )
        if optimal_batch_size < 0:
            raise ValueError(
                "brainbert_encoder state.optimal_batch_size must be non-negative."
            )

        self._optimal_batch_size = optimal_batch_size
        self.pool = str(state.get("pool", self.pool))
        self.clip_emb_k = int(state.get("clip_emb_k", self.clip_emb_k))
        self._state = {
            "feature_shape": [n_time, n_freq],
            "optimal_batch_size": optimal_batch_size,
            "pool": str(self.pool),
            "clip_emb_k": int(self.clip_emb_k),
        }

    def get_state(self):
        """Serialize the fitted feature-shape and batching state."""
        if self._state is None:
            return None
        return {
            "feature_shape": list(self._state["feature_shape"]),
            "optimal_batch_size": int(self._state["optimal_batch_size"]),
            "pool": str(self._state["pool"]),
            "clip_emb_k": int(self._state["clip_emb_k"]),
        }

    def reset_state(self):
        """Clear fitted split state so the next fold starts fresh."""
        self._state = None
        self._optimal_batch_size = None

    def _encode_sequences(self, sequences):
        """
        Encode per-channel STFT sequences.

        Args:
            sequences: Array/tensor of shape (n_sequences, n_time, n_freq).

        Returns:
            numpy.ndarray:
              - (n_sequences, hidden_dim) for mean/max pooling
              - (n_sequences, time, hidden_dim) for raw pooling
        """
        self._load_model()
        if isinstance(sequences, np.ndarray):
            data_reshaped = torch.from_numpy(sequences).float()
        elif torch.is_tensor(sequences):
            data_reshaped = sequences.float()
        else:
            data_reshaped = torch.as_tensor(sequences, dtype=torch.float32)

        if data_reshaped.ndim != 3:
            raise ValueError(
                "brainbert_encoder expects (n_sequences, n_time, n_freq), got "
                f"{tuple(data_reshaped.shape)}."
            )

        total_sequences, n_time, n_freq = data_reshaped.shape
        if self._state is not None:
            expected_n_time, expected_n_freq = (
                int(self._state["feature_shape"][0]),
                int(self._state["feature_shape"][1]),
            )
            if (n_time, n_freq) != (expected_n_time, expected_n_freq):
                raise ValueError(
                    "brainbert_encoder feature shape mismatch versus fitted state: "
                    f"expected {(expected_n_time, expected_n_freq)}, got {(n_time, n_freq)}."
                )

        if self._optimal_batch_size is None:
            self._optimal_batch_size = int(
                self._estimate_optimal_batch_size(n_time, n_freq)
            )
            if self.verbose:
                log(
                    f"[BrainBERT] Auto-detected batch size: {self._optimal_batch_size} "
                    f"(processing {total_sequences} sequences)",
                    priority=2,
                )

        optimal_batch = int(self._optimal_batch_size)
        if optimal_batch < 0:
            self._print_memory_error(n_time, n_freq, total_sequences)
            raise RuntimeError(
                f"Insufficient GPU memory. Cannot fit even minimum batch size {self.min_batch_size}. "
                f"See error details above."
            )

        optimal_batch = max(self.min_batch_size, optimal_batch)

        if total_sequences > optimal_batch:
            if self.verbose:
                n_chunks = (total_sequences + optimal_batch - 1) // optimal_batch
                log(
                    f"[BrainBERT] Processing in {n_chunks} chunks of {optimal_batch}",
                    priority=2,
                )
            outputs = self._process_in_chunks(data_reshaped, optimal_batch)
        else:
            data_reshaped = data_reshaped.to(self.device)
            with torch.no_grad():
                outputs = self.model(data_reshaped, src_key_mask=None)
            outputs = self._postprocess_encoder_output(outputs, n_time=n_time)

        return outputs.detach().cpu().numpy().astype(np.float32, copy=False)

    def transform_samples(self, samples):
        """Encode all channels across the sample list in one batched pass."""
        sample_list = list(samples)
        if not sample_list:
            return []
        if self._state is None:
            raise RuntimeError(
                "brainbert_encoder state is unset. Run fit_split(...) first."
            )

        expected_shape = tuple(int(v) for v in self._state["feature_shape"])
        channel_counts = []
        sequence_blocks = []
        for sample in sample_list:
            n_time, n_freq = self._extract_sample_feature_shape(sample)
            if (n_time, n_freq) != expected_shape:
                raise ValueError(
                    "brainbert_encoder feature shape mismatch versus fitted state: "
                    f"expected {expected_shape}, got {(n_time, n_freq)}."
                )
            x = np.asarray(sample["x"], dtype=np.float32)
            channel_counts.append(int(x.shape[0]))
            sequence_blocks.append(x)

        # Batch channels from every sample together so encoder work amortizes
        # across the whole split materialization call.
        all_sequences = np.concatenate(sequence_blocks, axis=0)
        encoded = self._encode_sequences(all_sequences)

        out_samples = []
        offset = 0
        for sample, n_channels in zip(sample_list, channel_counts):
            next_offset = offset + n_channels
            out = dict(sample)
            out["x"] = np.asarray(encoded[offset:next_offset], dtype=np.float32)
            out_samples.append(out)
            offset = next_offset
        return out_samples
