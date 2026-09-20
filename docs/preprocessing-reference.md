# Preprocessing reference

See the [README](../README.md#preprocessing) for config selection and leaderboard tracks.
See the [changelog](../CHANGELOG.md) for upgrade actions.

## Bundled recipes

- Configs live in [imindbench/conf/preprocessor/](../imindbench/conf/preprocessor/).
- `{rate}` is the input rate: 1000 Hz for BYD; 2048 Hz for NeuroprobeV2/PIPPI.
- All bundled recipes use notch filtering and Laplacian referencing.
- Filter defaults and omission/`null` behavior are documented beside the YAML fields.

| Config family (without `.yaml`) | Model inputs / normalization |
| --- | --- |
| `multi_stft_{rate}Hz` | Three STFT resolutions; training-fitted normalization per channel/frequency bin |
| `multi_stft_zscore_{rate}Hz` | Three STFT resolutions; per-sample, per-channel normalization |
| `stft_{rate}Hz` | Single-STFT representation |
| `stft_brainbert_{rate}Hz` | BrainBERT embeddings from its STFT and pretrained encoder |
| `wav_hpf_robust_{rate}to500Hz` | 500 Hz waveforms; 0.5 Hz high-pass; training-fitted robust scaling |
| `wav_hpf_zscore_{rate}to500Hz` | 500 Hz waveforms; 0.5 Hz high-pass; per-sample, per-channel z-score |
| `wav_nohpf_robust_{rate}to500Hz` | 500 Hz waveforms; no high-pass; training-fitted robust scaling |
| `wav_diver_{rate}to500Hz` | 500 Hz waveforms; DIVER filtering; input scaling handled by the model |
| `wav_barista_1000to2048Hz`, `wav_barista_2048Hz` | 2048 Hz waveforms; session-wise filtering; robust scaling then per-sample, per-channel z-score |

## Processing rules

- **Stage order:** `chain:` runs in order; each stage has its own `name:` and settings.
  A single-stage config uses `name:` directly.
- **Context:** 500 Hz waveform recipes filter with 15 seconds of context, then crop
  to the target window before rereferencing and resampling.
- **Resampling:** `resample` requires positive integer `source_rate` and `target_rate`
  in Hz. Crop context before changing rates; keep rate declarations consistent across stages.
- **Spectral inputs:** bundled STFTs use Torch at the native input rate, centered
  windows, reflection padding, and `padded: false`.
- **Normalization:** fit learned statistics on training data only and reuse them
  for validation/test data. Per-sample normalization is computed independently for each sample.
- **Multiple datasets:** `dataset.train_sources[].preprocessor` selects a preset by
  filename without `.yaml`; omission inherits the top-level pipeline.

## Window slicing

`dataset.window_slicing_policy` controls evaluation and context reads across all splits:

| Policy | Behavior |
| --- | --- |
| `ceil` (default) | Snap near-grid timestamps, then round both boundaries up |
| `legacy_floor` | Floor both boundaries relative to the recording's time origin, without snapping |

- Logs and result JSONs record the policy; preprocessing cache keys include it.
- `legacy_floor` reproduces earlier window reads, not unrelated processing/training changes.
- Use a fresh output root when changing settings; completed result JSONs are skipped.
