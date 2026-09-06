# Paper experiment coverage

Use the three examples in [EXPERIMENTS.md](EXPERIMENTS.md) to start. This catalog
maps paper experiment families to available recipes and their limitations.

## Experiment catalog

The three evaluation datasets below are NeuroprobeV2, BYD and PIPPI. Dataset
population, task and preprocessing matter as much as model selection.

| Paper family | Release workflow | Population / settings | Notes and remaining limits |
| --- | --- | --- | --- |
| Multi-STFT baselines: Table 1, Appendix 2 | `paper_multistft` | Three datasets; Logistic/MLP/CNN/PopT; within-session; 1000 Hz BYD, 2048 Hz others; PIPPI high-cov | Explicit model tolerances, PopT learning rates, training caps and worker settings. See temporal window limits below. |
| BrainBERT-STFT baseline classifiers: Table 1 | `paper_brainbert_stft` | Three datasets; Logistic/MLP/CNN; single STFT with 40-bin cutoff; BYD overlap 0.75 | No encoder checkpoint required for these classifiers. |
| BrainBERT encoder + linear readout | `brainbert` | Three datasets; within-session; encoder resource required | Distinct from standalone STFT classifiers; full historical launch mapping remains incomplete. |
| HTNet 500 Hz waveform: Table 1, Appendix 1/2 | `paper_htnet500` | Three datasets; 15-second context, high-pass filtering, crop, Laplacian, downsample, global robust scaling | Includes the rebuttal baseline deterministic setting. Other HTNet populations/profiles must be matched separately. |
| DIVER waveform: Table 1, Appendix 2 | `paper_diver` | Three datasets; DIVER filter, 15-second context, 500 Hz; frozen encoder; PIPPI full | Requires an external checkpoint and writable model directory; see resource requirements below. |
| Other waveform baselines / BaRISTA | `barista`; model-specific retained waveform presets | Small BaRISTA example population; session filtering/scaling and model-specific metadata | The full set of waveform experiments is not yet mapped to recipes. |
| STFT sweep: Figure 4 and STFT appendices | `stft_sweep` | Three datasets; Logistic; 3 windows × 3 overlaps × 4 frequency ceilings | Temporal slicing and runtime profile depend on the historical result family. |
| Logistic STFT variants / preprocessing baselines | Retained single/multi-STFT and waveform presets; custom recipe copies | High-frequency, z-score, sample/session normalization are separate scientific variations | Partial mapping; some variants require overrides. Output names alone do not establish equivalent settings. |
| Sample efficiency: Figure 3 / Appendix 4 | `sample_efficiency` | NeuroprobeV2; Logistic/MLP/CNN/PopT; fractions 1, 1/2, 1/4, 1/8, 1/16 | Fraction workflow; exact historical launches and older hold-in variants are not all covered. |
| Hold-in / multisource: Figure 3 / Appendix 5 | `hold_in`, `multisource` | Three datasets; PopT; selected decodable populations; within-session or hold-in-session | Match the panel population, output family and temporal window policy. |
| Processed-data figures: Figure 2 / Appendix 6 | TorchBrain prepared data and retained Neuroprobe2025 view | Prepared recording/channel metadata; Appendix 6 multi-STFT path | Figure rendering is not included; Figure 4b also requires manual assembly. |

## Using the paper recipes

The four `paper_*` recipes specify model, preprocessing, training and runtime
settings for the corresponding paper experiment families.

These grids use the release task/target catalog in `recipes/datasets.yaml`.
They do not recover every historical per-task change, exact paper population or
checkpoint. Select the relevant targets/tasks explicitly; match the population
manifest used by the panel before aggregating new results.

Use the same `imindbench-grid` interface with `--recipe paper_multistft`,
`paper_brainbert_stft`, `paper_htnet500` or `paper_diver`. Default behavior previews
commands. Set required checkpoint paths with `--set KEY=VALUE`; see
[data and checkpoints](EXPERIMENTS.md#data-and-checkpoints) for resource settings
and availability.

Historical configs can differ from current model defaults. The recipes preserve
those differences as explicit overrides, including per-model training caps.
Dataset/model identity and output routing cannot be changed through these
per-model overrides. Caller `--set` tuning takes precedence and creates a new
configuration; use a fresh output root when changing it.

## Configuration scope

- **Paper implementations:** all eight model families and scientifically
  distinct filter, rate, context, crop, normalization and STFT operations.
- **Advanced:** region-pooling presets support aligned-channel cross-subject
  models beyond the introductory examples.
- **Compatibility:** Neuroprobe2025 and the direct Laplacian-STFT class remain
  for figure inputs and external historical configs.

## Reproducing results

Select the recipe, dataset, tasks and target population for the experiment.
Retain the exact command and resolved configuration, dataset version and
preparation settings, population manifest, random seed, package versions and
checkpoint hash. Record CPU/GPU details for torch models. Use a separate external
output directory for each configuration. GPU training may vary across devices
and software versions even with a fixed seed.

Earlier BYD/PIPPI experiments used floor-based temporal window boundaries.
Public TorchBrain uses newer slicing semantics; this release does not expose
a legacy-floor option. PopT coordinate indexing is a separate setting and does
not control temporal slicing. Default commands therefore do not guarantee exact
numerical agreement with every earlier result.

Recipes specify settings but do not bundle data, checkpoints or paper result
files. Match the panel's population and per-run settings before aggregating
results. Checkpoint requirements and known availability gaps are documented in
[EXPERIMENTS.md](EXPERIMENTS.md#data-and-checkpoints).

Full paper-figure rendering is not included. Regenerating panels also requires
existing result JSONs and explicit panel populations; data-summary figures need
prepared recording/channel metadata. Figure 4b includes manual assembly.
