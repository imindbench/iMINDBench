# Paper experiment coverage

Use the three examples in [EXPERIMENTS.md](EXPERIMENTS.md) to start. This catalog
separates useful release workflows from recovered historical settings. All model
families used by the scoped paper notebooks remain available.

## Experiment catalog

The three evaluation datasets below are NeuroprobeV2, BYD and PIPPI. Dataset
population, task and preprocessing matter as much as model selection.

| Paper family | Release workflow | Population / settings | Evidence and remaining limits |
| --- | --- | --- | --- |
| Multi-STFT baselines: Table 1, Appendix 2 | `paper_multistft` | Three datasets; Logistic/MLP/CNN/PopT; within-session; 1000 Hz BYD, 2048 Hz others; PIPPI high-cov | Saved model tolerances, PopT learning rates, caps and historical worker profile recovered. BYD/PIPPI historical floor windows remain unavailable in the normal CLI. |
| BrainBERT-STFT baseline classifiers: Table 1 | `paper_brainbert_stft` | Three datasets; Logistic/MLP/CNN; single STFT with 40-bin cutoff; BYD overlap 0.75 | Saved classifier/preprocessor settings recovered. No encoder checkpoint required for these classifiers. |
| BrainBERT encoder + linear readout | `brainbert` | Three datasets; within-session; encoder resource required | Original launcher structure retained; exact historical checkpoint and full launch mapping unresolved. Distinct from standalone STFT classifiers. |
| HTNet 500 Hz waveform: Table 1, Appendix 1/2 | `paper_htnet500` | Three datasets; 15-second context, high-pass filtering, crop, Laplacian, downsample, global robust scaling | Representative rebuttal baseline config recovered, including deterministic setting. Other dated HTNet populations/profiles must be matched separately. |
| DIVER waveform: Table 1, Appendix 2 | `paper_diver` | Three datasets; DIVER filter, 15-second context, 500 Hz; frozen encoder; PIPPI full | Representative saved settings recovered. Checkpoint identity/retrieval unresolved; caller must supply checkpoint and writable model directory. |
| Other waveform baselines / BaRISTA | `barista`; model-specific retained waveform presets | Small BaRISTA example population; session filtering/scaling and model-specific metadata | Launcher structure retained. Selected BaRISTA checkpoint identity and complete waveform family mapping unresolved. |
| STFT sweep: Figure 4 and STFT appendices | `stft_sweep` | Three datasets; Logistic; 3 windows × 3 overlaps × 4 frequency ceilings | Existing launcher selections retained; legacy slicing/profile depends on the historical result family. |
| Logistic STFT variants / preprocessing baselines | Retained single/multi-STFT and waveform presets; custom recipe copies | High-frequency, z-score, sample/session normalization are separate scientific variations | Partial mapping. Some historical variants were expressed as overrides; do not infer equivalence from output names or add duplicate presets solely to match names. |
| Sample efficiency: Figure 3 / Appendix 4 | `sample_efficiency` | NeuroprobeV2; Logistic/MLP/CNN/PopT; fractions 1, 1/2, 1/4, 1/8, 1/16 | Explicit release workflow, not recovered exact historical launcher; older hold-in variants are not all covered. |
| Hold-in / multisource: Figure 3 / Appendix 5 | `hold_in`, `multisource` | Three datasets; PopT; selected decodable populations; within-session or hold-in-session | Retained launcher structure. Original checkpoint, population, output family and historical window policy still determine comparability. |
| Processed-data figures: Figure 2 / Appendix 6 | TorchBrain prepared data and retained Neuroprobe2025 view | Prepared recording/channel metadata; Appendix 6 multi-STFT path | Rendering not ported. Keep the Neuroprobe2025 config; Figure 4b manual assembly and Appendix 6 rendering caveats remain. |

## Recovered recipes

The four `paper_*` recipes recover scientific fields from **27 representative
saved configs**, one per selected dataset/model/result family. Their hashes and
sanitized scientific fields are preserved in
[`tests/paper_recipe_reference.json`](../tests/paper_recipe_reference.json).
Tests compose the real Hydra configs and compare those fields. This is config
recovery evidence, not a new numerical parity experiment.

These grids use the release task/target catalog in `recipes/datasets.yaml`.
They do not recover every historical per-task change, exact paper population or
checkpoint. Select the relevant targets/tasks explicitly; match the population
manifest used by the panel before aggregating new results. New output groups
start with `paper_` and never point into the historical result tree.

Use the same `imindbench-grid` interface with `--recipe paper_multistft`,
`paper_brainbert_stft`, `paper_htnet500` or `paper_diver`. Default behavior previews
commands. For DIVER, supply `--set model.upstream_ckpt=/path/to/checkpoint` and
`--set model.model_dir=/path/to/writable/shapes` before execution. Resource
requirements and retrieval gaps are in [RESOURCES.md](RESOURCES.md).

Historical configs can differ from current model defaults. The recipes preserve
those differences as explicit overrides, including per-model training caps.
Dataset/model identity and output routing cannot be changed through these
per-model overrides. Caller `--set` tuning takes precedence and creates a new
configuration; use a fresh output root when changing it.

## Configuration scope

- **Paper implementations:** keep all eight model families and scientifically
  distinct filter, rate, context, crop, normalization and STFT operations.
- **Advanced:** region-pooling presets support aligned-channel cross-subject
  models. They are outside the introductory examples, not unused functionality.
- **Compatibility:** Neuroprobe2025 and the direct Laplacian-STFT class remain
  for figure inputs and external historical configs.
- **Archived presets:** `laplacian_stft_1000Hz_2048Hz` and
  `laplacian_wav_HPF_sample_per_channel_time_1000Hz_1024Hz` had no name or exact
  chain matches across 90,526 saved paper-tree configs. Their original YAMLs
  remain in maintainer provenance, outside the installed package. The shared
  resampler and 36 other preprocessor presets remain.

Long migration narratives are archived under `provenance/history/` in the
maintained repository. The original notebook/config hashes remain historical
records; this catalog does not silently replace them.

## Reproduction boundary

Existing selected-case parity evidence is sufficient for this cleanup. No
training, experiment matrix expansion or notebook execution is needed to verify
these code and packaging changes. See [REPRODUCTION.md](REPRODUCTION.md) for the
legacy-window limitation and verified versus unresolved checkpoint cases.

Paper-panel regeneration is a separate remaining capability. A future compact
analysis entrypoint should consume existing result JSONs and explicit panel
manifests; data-summary figures also need prepared metadata. The release does
not currently claim one-command regeneration of all paper figures.
