# Experiment recipes

`imindbench-grid` is the single grid entrypoint. It expands a named YAML recipe
into ordinary `imindbench.run_eval` commands. By default it only prints commands;
add `--execute` to run them serially. No Git checkout or writable installation is
required. `python -m imindbench.launch` exposes the same interface.

Choose the canonical dataset name: `neuroprobev2`, `kelesbyd2024`, or
`berezutskayapippi2022`. Machine settings live in an external `paths/local.yaml`
created from `imindbench/conf/paths/example.yaml`.

## Three starting examples

Each example previews one evaluation. Inspect the command, then add `--execute`
when the data and any checkpoint are available. Use a distinct external output
root for each experiment configuration.

The source checkout and submission archive include three Bash launchers:

| Script | Evaluation | Setup |
| --- | --- | --- |
| `scripts/run_logistic.sh` | CPU Logistic | Base install |
| `scripts/run_mlp.sh` | GPU MLP | Base install and a CUDA-capable runtime |
| `scripts/run_barista.sh` | GPU BaRISTA | BaRISTA dependencies and checkpoint below |

Run from the source root, replacing the script name as needed:

```bash
bash scripts/run_mlp.sh \
  --config-dir /path/to/config --output-root /path/to/runs/mlp
```

Add `--execute` to run or `--execute --resume` to resume. All three select
NeuroprobeV2, onset and `sub1_sess1`. They use the active Python environment and
forward arguments to the grid launcher. For example, append `--device cuda:1`
to select another GPU or `--set model.max_iter=5` for a shorter trial. Use a new
output root when changing settings.

Model/task/target flags select lists, so repeating them adds selections. To
replace these selections, edit a caller-owned copy of the example or use
`imindbench-grid` directly with the recipe and selections you want. Shell scripts
are source examples; wheel installations provide `imindbench-grid` directly.
You can also copy the scripts elsewhere and invoke them by absolute path.

For BaRISTA, set `paths.barista_checkpoint` in your external paths file or append
`--set paths.barista_checkpoint=/path/to/barista.ckpt`.

The BaRISTA recipe uses a small target/task subset and sets explicit
learning rates, worker settings and scheduler overrides.

## Optional models

The README's base install is sufficient for Logistic, MLP, CNN and HTNet.
For all other retained model dependencies, install the paper extra from the
source root:

```bash
python -m pip install '.[paper]'
```

This installs dependencies, not weights. It includes a pinned warmup scheduler
that requires Git and network access. To install only what a particular model
needs, use one of these extras instead:

| Extra | Models |
| --- | --- |
| `.[warmup]` | PopT and BrainBERT linear readout |
| `.[barista,warmup]` | BaRISTA |
| `.[diver]` | DIVER |
| `.[logging]` | Optional Weights & Biases logging |

## Other datasets

Prepare BYD or PIPPI with the same public TorchBrain installation:

```bash
brainsets prepare keles_byd_2024 --raw-dir /path/to/byd/raw --processed-dir /path/to/processed
brainsets prepare berezutskaya_pippi_2022 --raw-dir /path/to/pippi/raw --processed-dir /path/to/processed
```

Review each pipeline's storage requirements, dataset terms and credentials
before downloading. Select `kelesbyd2024` or `berezutskayapippi2022` through
`imindbench-grid --dataset`; use the appropriate dataset directory in your paths
configuration. Neuroprobe2025 remains available through `imindbench` for
historical splits and data figures; the examples use NeuroprobeV2.

## Data and checkpoints

Prepare recordings following the [quickstart](../README.md#prepare-public-data).
Use absolute paths in your external paths config and keep data, checkpoints,
caches and outputs outside the installation.

| Model or preprocessing path | Required resources |
| --- | --- |
| Logistic, MLP, CNN, HTNet | No pretrained checkpoint |
| BrainBERT encoder + linear readout | `paths.brainbert_checkpoint` |
| PopT | `paths.popt_checkpoint`; numerical channel coordinates from the provider |
| BaRISTA | `paths.barista_checkpoint`; coordinate/Destrieux metadata; compatible xformers runtime |
| DIVER | `model.upstream_ckpt` and writable `model.model_dir`; matching coordinate and waveform config |

Checkpoints are not bundled, and public retrieval instructions are not yet
documented here for BrainBERT, PopT, BaRISTA or DIVER. The selected paper
checkpoint identities remain unresolved for BrainBERT, BaRISTA and DIVER.
The main-results PopT checkpoint has SHA256
`cf4e835d5309559d468b2f1ebd9b76882398c30bedae6c0c8bc6fdb4c506b52f`;
other PopT experiment families may use different weights. This hash identifies
an artifact, not a download location, and the missing instructions do not imply
that upstream projects lack published weights.

Use the checkpoint required by your experiment and record its SHA256;
substituting weights from the same model family can change the results.
Start with Logistic, MLP, CNN or HTNet if the required checkpoint is unavailable.

## Recipe catalog

| Recipe | What it selects |
| --- | --- |
| `baselines` | Within-session; Neuroprobe Logistic/MLP/CNN/PopT; BYD/PIPPI PopT and native-rate HTNet |
| `brainbert` | Within-session BrainBERT encoder + linear readout |
| `barista` | Three tasks × two targets per dataset |
| `stft_sweep` | Logistic; 3 windows × 3 overlaps × 4 frequency ceilings |
| `sample_efficiency` | Neuroprobe Logistic/MLP/CNN/PopT, fractions 1 through 1/16 |
| `hold_in` | PopT hold-in-session with packaged decodable population |
| `multisource` | PopT with three-provider training, within-session evaluation |
| `paper_multistft` | Multi-STFT Logistic/MLP/CNN/PopT on three datasets |
| `paper_brainbert_stft` | Single-STFT Logistic/MLP/CNN on three datasets; no encoder checkpoint |
| `paper_htnet500` | HTNet with 15-second waveform context at 500 Hz on three datasets |
| `paper_diver` | Frozen DIVER encoder with 15-second waveform context at 500 Hz on three datasets |

Use `baselines` for an introductory run. See
[PAPER_COVERAGE.md](PAPER_COVERAGE.md) for the relationship between these recipes
and paper experiments, including population and reproduction limits.

Use repeated `--model`, `--task`, `--regime`, and `--target` flags to select a
subset already present in a recipe. `--count` prints the count after filtering;
`--limit N` selects the first N jobs in deterministic recipe order. Hold-in and
multisource use the packaged `stft_or_htnet_500hz_val_mean0p60` population for
both eligible targets and training; `--decodable-dir` selects a custom manifest
directory. Dataset splits remain owned by the selected TorchBrain provider.

## Configuration variations

Recipes live in `imindbench/recipes/`; `datasets.yaml` supplies shared tasks and
named target lists. Copy a recipe to a caller-owned YAML file and pass its path
with `--recipe` to change model selections, preprocessors or sweep dimensions.
Dataset-specific fields override recipe-wide `overrides`. Model entries may
select a different preprocessor and override scalar model/dataset settings
without changing identity or subset. Per-model settings override dataset-level
settings, and caller `--set` tuning takes precedence. Sweep values are numeric Hydra scalar strings;
labels form distinct output subdirectories.

Use `--set KEY=VALUE` for ordinary Hydra settings such as checkpoint paths,
learning rates or logging. The launcher rejects overrides of identity fields,
output directories and sweep dimensions, since those determine target filtering
and output naming. Change those through recipe fields and selection flags.
A custom recipe using an incompatible model/preprocessor combination will still
fail the evaluation's runtime validation; previewing commands does not read data
or establish scientific compatibility.

## Outputs and resume

Each evaluation writes its normal Hydra files and `population_*.json`, plus
`launch.json` (exact argv), `launcher.log` and a `completed.sha256` marker after
successful completion and JSON parsing. This is operational completion evidence,
not proof of checkpoint/environment/data identity or numerical reproduction.

The launcher locks the output root while executing. A second launcher using the
same root fails immediately. Use separate roots for independent GPU workers.
There is no machine-specific GPU selection or artificial startup delay. Standard
`TMPDIR` controls temporary storage; keep it short and external to the package.

`--execute --resume` accepts only directories with the same saved command.
Completed results must match their completion hash to be skipped. Failed jobs
without results can be retried; changed/unverified result files are rejected.
Results without a completion record cannot be resumed; use a fresh root. Changing inputs in place is not detected by command
identity, so keep inputs immutable and retain their hashes separately.
