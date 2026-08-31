# iMINDBench

iMINDBench is a standalone, Hydra-configured evaluation package for public
intracranial neural decoding datasets. It consumes processed datasets from the
public `torch_brain` package and writes one portable result JSON per evaluation.

This repository retains the legacy paper model families—Linear/BrainBERT
baseline, Logistic, MLP, CNN, HTNet, PopT, and DIVER—plus BaRISTA. It does not
include MVPFormer, SEEGnificant, or leaderboard/submission services. Use
`imindbench --help` to see the exact installed Hydra groups; the presence of a
group does not by itself establish paper parity for a particular run.

## Install

The canonical environment is Python 3.10 in the root
[`environment.yml`](../environment.yml). From the repository root:

```bash
conda env create -f environment.yml
conda activate imindbench
python -m pip install "torch_brain @ git+https://github.com/neuro-galaxy/torch_brain.git@492f94a594e81d30ef38db32d8be145627421b0d"
python -m pip install -e .
python -m pip check
imindbench --help
```

`torch_brain` is installed separately at an immutable commit because it owns the
dataset loaders and the merged `brainsets` CLI. Do not install the former
standalone `brainsets` package. Do not use the `torch_brain` development extra;
the iMINDBench environment owns the pinned PyTorch stack.

For a release deployment, replace the development pin above only with a reviewed
immutable release or commit. An editable local `torch_brain-public` checkout is
appropriate for development, but is not a reproducible public installation.

## Prepare public data

Keep raw data, processed H5 files, caches, and run outputs outside this Git
checkout. The high-level preparation flow is:

```bash
brainsets list
brainsets prepare neuroprobe_2025 --raw-dir /path/to/raw --processed-dir /path/to/processed
brainsets prepare keles_byd_2024 --raw-dir /path/to/raw --processed-dir /path/to/processed
brainsets prepare berezutskaya_pippi_2022 --raw-dir /path/to/raw --processed-dir /path/to/processed
```

Preparation may download large datasets and uses each pipeline's isolated
environment. Review `brainsets prepare --help`, storage requirements, dataset
terms, and any required credentials before starting. NeuroprobeV2 is an
alternate split/recording view over the artifacts produced by
`brainsets prepare neuroprobe_2025`; there is no separate NeuroprobeV2 prepare
command.

Copy [`conf/paths/example.yaml`](conf/paths/example.yaml) to a machine-local
`conf/paths/server_<name>.yaml`, set `dataset_root` to the processed root, and
select it with `paths=server_<name>`. These `server_*.yaml` files are ignored by
Git. Add only the checkpoint and cache paths required by the selected model and
runtime.

## Run an evaluation

The installed command is the canonical entrypoint. A CPU Logistic run has this
shape:

```bash
imindbench \
  paths=server_local \
  dataset=neuroprobe2025 \
  dataset.regime=SS-SM \
  dataset.task=onset \
  dataset.test_subject=1 \
  dataset.test_session=1 \
  model=logistic \
  preprocessor=laplacian_stft_2048Hz \
  wandb.enabled=false \
  runtime.overwrite=false \
  hydra.run.dir=/path/to/runs/logistic-onset-sub1-session1
```

The equivalent low-level entrypoint is:

```bash
python -m imindbench.run_eval --help
```

Replace `imindbench` with `python -m imindbench.run_eval` to use the same Hydra
arguments through the low-level entrypoint.

Hydra writes `.hydra/`, logs, and `population_*.json` into `hydra.run.dir`.
Give every run a distinct output directory outside the source tree and keep
`runtime.overwrite=false` unless replacement is deliberate. Do not point
multiple simultaneous runs at the same directory.

Neuroprobe2025 uses `SS-SM`, `SS-DM`, and `DS-DM`. NeuroprobeV2, BYD, and PIPPI
use the configured within/hold-in/hold-out regimes exposed by their dataset
groups. Model/preprocessor compatibility, coordinates, sampling rate, and
checkpoint requirements are validated at runtime; start from a shipped matching
config rather than mixing groups blindly.

### CPU and GPU gates

- Installation, both help entrypoints, config composition, and a small Logistic
  run are the CPU acceptance gate.
- Before a CUDA model run, require `torch.cuda.is_available()` in the active
  environment and use a device override supported by the selected model.
- Checkpoint-backed PopT and BaRISTA runs require caller-supplied checkpoints.
  BaRISTA also requires `xformers`, compatible CUDA/PyTorch builds for GPU use,
  channel coordinates, and appropriate Destrieux region metadata. There is no
  bundled BaRISTA checkpoint.
- A CPU smoke result does not substitute for the final GPU acceptance run for a
  GPU-targeted model.

## Paper-reference parity tools

[`scripts/parity_tools.py`](../scripts/parity_tools.py) is an offline tool. It
constructs commands but never launches them, and it compares existing result
JSONs against the five records in
[`artifacts/parity_reference/`](../artifacts/parity_reference/). Inspect its
validated interfaces with:

```bash
python scripts/parity_tools.py build-commands --help
python scripts/parity_tools.py compare --help
```

Both modes require caller-owned JSON mappings. `checkpoint-map.json` maps the
logical keys `popt_checkpoint` and/or `barista_checkpoint` to files when a
selected case needs them. `resource-map.json` maps auxiliary resource keys; an
empty object is valid when the selected case needs none. Comparison additionally
uses `candidate-map.json`, mapping case IDs to existing `population_*.json`
files. Run and report output roots must not already exist.
For the checkpoint-free Logistic example below, both `checkpoint-map.json` and
`resource-map.json` may contain the single JSON object `{}`.

Example dry-run construction for the checkpoint-free Logistic case:

```bash
python scripts/parity_tools.py \
  --manifest artifacts/parity_reference/manifest.json \
  build-commands \
  --case neuroprobev2_logistic_multistft_onset_sub1_sess1 \
  --data-root /path/to/processed \
  --output-root /path/to/fresh-run-root \
  --checkpoint-map checkpoint-map.json \
  --resource-map resource-map.json
```

Example comparison:

```bash
python scripts/parity_tools.py \
  --manifest artifacts/parity_reference/manifest.json \
  compare \
  --case neuroprobev2_logistic_multistft_onset_sub1_sess1 \
  --candidate-map candidate-map.json \
  --checkpoint-map checkpoint-map.json \
  --resource-map resource-map.json \
  --report-dir /path/to/fresh-report-root
```

`PASS` means metric/config-record parity only; it does not prove execution
provenance. `FAIL` and `MISSING` are nonzero exits. PopT and BaRISTA references
remain `NOT-COMPARABLE`—also a nonzero exit—when every checkable field matches
but the exact historical checkpoint hash is unknown. See the
[`parity reference README`](../artifacts/parity_reference/README.md) for the
precise status contract.

## Licensing

The repository is Apache-2.0 except for identified third-party components. In
particular, BaRISTA is under the USC educational, research, and non-profit
license; commercial use requires separate USC permission. Review
[`THIRD_PARTY.md`](../THIRD_PARTY.md) and the
[`BaRISTA license`](../LICENSES/BaRISTA-LICENSE.md) before redistribution or use.
