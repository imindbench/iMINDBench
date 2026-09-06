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

CPU Logistic:

```bash
imindbench-grid --recipe baselines --dataset neuroprobev2 \
  --model logistic --task onset --target sub1_sess1 --device cpu \
  --config-dir /path/to/config --paths local --output-root /path/to/runs/cpu
```

GPU MLP, without a pretrained checkpoint:

```bash
imindbench-grid --recipe baselines --dataset neuroprobev2 \
  --model mlp --task onset --target sub1_sess1 --device cuda:0 \
  --config-dir /path/to/config --paths local --output-root /path/to/runs/mlp
```

Checkpoint-backed BaRISTA:

```bash
imindbench-grid --recipe barista --dataset neuroprobev2 \
  --task onset --target sub1_sess1 --device cuda:0 \
  --config-dir /path/to/config --paths local --output-root /path/to/runs/barista \
  --set paths.barista_checkpoint=/path/to/barista.ckpt
```

The BaRISTA recipe uses a small target/task subset and sets explicit
learning rates, worker settings and scheduler overrides. The exact paper checkpoint
identity and retrieval instructions remain incomplete.

## Recipe catalog and paper coverage

| Recipe | What it selects | Paper relationship |
| --- | --- | --- |
| `baselines` | Within-session; Neuroprobe Logistic/MLP/CNN/PopT; BYD/PIPPI PopT and native-rate HTNet | Partial Table 1/Appendix 2 coverage; selected model examples |
| `brainbert` | Within-session BrainBERT encoder + linear readout | BrainBERT row; does not cover the three standalone STFT classifiers |
| `barista` | Three tasks × two targets per dataset | BaRISTA command structure; historical checkpoint/launcher mapping incomplete |
| `stft_sweep` | Logistic; 3 windows × 3 overlaps × 4 frequency ceilings | Figure 4 / STFT appendices |
| `sample_efficiency` | Neuroprobe Logistic/MLP/CNN/PopT, fractions 1 through 1/16 | Explicit fraction workflow; not a recovered historical launcher |
| `hold_in` | PopT hold-in-session with packaged decodable population | Figure 3 / Appendix 5 family |
| `multisource` | PopT with three-provider training, within-session evaluation | Figure 3 / Appendix 5 family |

Four additional `paper_*` recipes recover representative multi-STFT,
BrainBERT-STFT classifier, HTNet 500 Hz and DIVER settings from saved configs.
See [PAPER_COVERAGE.md](PAPER_COVERAGE.md) for exact scope and evidence. Use
`baselines` for an introductory run and a paper recipe when those recovered
settings are intended. All paper model families remain available; two unused rate-conversion presets
have been archived as described in the coverage guide. Not every paper variant
has a complete grid.

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
