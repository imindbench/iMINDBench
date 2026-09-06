# Running experiments

The Bash scripts select experiments and call a shared Python launcher for
preview, execution and resume. Model, dataset, preprocessing and named
experiment settings use ordinary Hydra configs under `imindbench/conf/`.
`imindbench-grid` and `python -m imindbench.launch` expose the same execution
interface for custom scripts and wheel installations. No Git checkout or
writable installation is required.

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

Use `--task onset speech` or `--target sub1_sess1 sub3_sess0` to replace the
example's task or target selection. The last occurrence of either flag wins.
Shell scripts are source examples; wheel installations provide `imindbench-grid`
directly. You can copy the scripts elsewhere and invoke them by absolute path.

For BaRISTA, set `paths.barista_checkpoint` in your external paths file or append
`--set paths.barista_checkpoint=/path/to/barista.ckpt`.

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

## Experiment families

Use one script for larger grids:

```bash
bash scripts/run_experiments.sh paper_multistft neuroprobev2 mlp \
  --config-dir /path/to/config --output-root /path/to/runs/paper_mlp
```

Arguments are `FAMILY DATASET [MODEL|all]`, followed by launcher options. Omit the
model, or use `all`, to select every model listed by that family. Models run
serially, and the script stops if a model fails. Add `--count` to print one count
per model, or `--limit N` to preview/run at most N evaluations **per model**.
Use `--task` and `--target` to narrow the grid before counting or execution.

| Family | Selections |
| --- | --- |
| `baselines` | Neuroprobe Logistic/MLP/CNN/PopT; BYD/PIPPI PopT and native-rate HTNet |
| `brainbert` | BrainBERT encoder + linear readout; all 15 tasks and 5 / 29 / 5 subject/session pairs |
| `barista` | All 15 tasks; 5 NeuroprobeV2, 29 BYD or 5 PIPPI subject/session pairs |
| `stft_sweep` | Logistic; 3 windows × 3 overlaps × 4 frequency ceilings |
| `sample_efficiency` | Neuroprobe Logistic/MLP/CNN/PopT, fractions 1 through 1/16 |
| `hold_in` | PopT hold-in-session with the decodable population |
| `multisource` | PopT with three-provider training and within-session evaluation |
| `paper_multistft` | Multi-STFT Logistic/MLP/CNN/PopT on three datasets |
| `paper_brainbert_stft` | Single-STFT Logistic/MLP/CNN on three datasets; no encoder checkpoint |
| `paper_htnet500` | HTNet with 15-second waveform context at 500 Hz |
| `paper_diver` | Frozen DIVER encoder with 15-second waveform context at 500 Hz |

[Paper coverage](PAPER_COVERAGE.md) maps these families to paper experiments.
Shared tasks and target sets live in `conf/units/catalog.yaml`; `--unit-set`
selects a named set. Hold-in and
multisource use `stft_or_htnet_500hz_val_mean0p60` for eligible targets and training;
`--decodable-dir` selects a custom manifest directory instead. Tasks with no
eligible targets are skipped. Dataset splits belong to the TorchBrain provider.

## Custom experiments

Edit a caller-owned copy of a shell script to change model/preprocessor
selections or sweep values. Training settings live in `conf/experiment/`, as
native Hydra overrides of `model`, `dataset`, `runner` and `runtime`. For example,
`paper_multistft/mlp.yaml` includes shared settings and specifies the MLP tolerance.
These configs also work with the single-evaluation CLI as
`experiment=paper_multistft/mlp`, alongside dataset/model/preprocessor selections.

To define a new experiment, put a YAML file in your external config directory:

```yaml
# <config-dir>/experiment/my_trial.yaml
# @package _global_
model:
  max_iter: 20
  learning_rate: 0.001
```

A custom grid can select it directly:

```bash
imindbench-grid --dataset neuroprobev2 --model mlp \
  --preprocessor laplacian_multi_stft_2048Hz --experiment my_trial \
  --task onset --target sub1_sess1 --device cuda:0 \
  --config-dir /path/to/config --paths local --output-root /path/to/runs/my_trial
```

Use `--set KEY=VALUE` for tuning. These values override the experiment config;
when a key is repeated, the last value wins. Identity fields such as model,
dataset, task, subset and output routing use explicit selection flags. For
parameter grids, add `--sweep model.learning_rate=0.001,0.0001`; repeated `--sweep`
flags form a Cartesian product. Sweep keys must be distinct from `--set` keys,
and sweep values must be numeric scalars. Change the built-in sweep dimensions
in a script copy rather than adding a second sweep for the same key.

The family script fixes dataset/model/preprocessor/experiment selection through
its positional arguments. For different combinations, use a custom script or
the shared launcher directly. Configuration composition does not load data or
establish scientific compatibility; runtime validation still applies.

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

The family scripts group outputs by family/dataset, model/preprocessor, subset,
sweep settings, regime, task and target. Use a fresh output root when changing
experiment settings or switching from the former recipe-based commands: their
saved command records differ, so they cannot be resumed with these commands.
