# iMINDBench

An independently installable benchmark for intracranial neural decoding.
Public TorchBrain provides data preparation and dataset loaders; iMINDBench
provides preprocessing, models, and evaluation.

## Install

Use Python 3.10. From the repository root:

```bash
conda env create -f environment.yml
conda activate imindbench
python -m pip install "torch_brain @ git+https://github.com/neuro-galaxy/torch_brain.git@e39f48ce0ec8c8f59be2507dca8ae172cce79d28"
python -m pip install .
python -m pip check
imindbench --help
```

The base install supports Logistic, MLP, CNN and HTNet. Torch is required even
for CPU preprocessing. Specialized dependencies are opt-in:

| Extra | Use |
| --- | --- |
| `.[warmup]` | PopT or BrainBERT linear readout with the configured warmup scheduler |
| `.[barista,warmup]` | BaRISTA, including its scheduler |
| `.[diver]` | DIVER |
| `.[paper]` | All retained model dependencies at the reviewed version constraints |
| `.[logging]` | Optional Weights & Biases logging |

For example, install the complete paper stack with `python -m pip install '.[paper]'`.
This adds dependencies, not pretrained weights. The warmup extra retains its
pinned Git source, so that extra still needs Git and network access. A submission
bundle can provide TorchBrain as a local wheel; follow its included install step.

TorchBrain is installed separately at the reviewed immutable commit. It owns
`brainsets` and the dataset loaders; do not install the former standalone
BrainSets package. Avoid the TorchBrain development extra, which has a separate
training stack. Dependency constraints live in `pyproject.toml`;
`environment.yml` only creates the Python environment.

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

Create a machine-local configuration directory outside the installation, with
`paths/local.yaml` copied from [`imindbench/conf/paths/example.yaml`](imindbench/conf/paths/example.yaml).
Set its `dataset_root` and any required checkpoint/cache paths to absolute paths.
Pass `--config-dir /path/to/config paths=local` to the CLI. This works with both
source and wheel installs and keeps private settings out of the package.

## First evaluation

Preview one checkpoint-free NeuroprobeV2 Logistic evaluation on CPU:

```bash
imindbench-grid --recipe baselines --dataset neuroprobev2 \
  --model logistic --task onset --target sub1_sess1 --device cpu \
  --config-dir /path/to/config --paths local --output-root /path/to/runs/logistic
```

Add `--execute` to run it after preparing data and setting the external paths.
The default is a dry run. Outputs include `population_*.json`, the resolved Hydra
config, launch command and log under your output root. Reuse completed grid
outputs only through the documented `--resume` behavior.

For one custom configuration, use `imindbench` (equivalently
`python -m imindbench.run_eval`) with Hydra overrides. NeuroprobeV2 is the default
view; Neuroprobe2025 remains available for historical splits and data figures.
Remote logging is disabled by default. Author and organization metadata are
optional and default to null in result JSONs.

## Experiments and reproduction

- [Experiments](docs/EXPERIMENTS.md): three examples, checkpoints, recipes and launcher options.
- [Paper coverage](docs/PAPER_COVERAGE.md): experiment mapping and reproduction limits.

All eight model families are retained: Logistic, Linear/BrainBERT, MLP, CNN,
HTNet, PopT, DIVER and BaRISTA. Exact reproduction depends on the dataset
version, preprocessing, runtime settings and checkpoint used.
Data and pretrained weights are not bundled.

## Development

```bash
python -m pip install -e '.[dev]'
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Install the documented TorchBrain dependency first. Optional-model tests require
the corresponding extras.

Build a source archive from the repository root:

```bash
python scripts/build_submission.py --output /path/to/imindbench-submission.tar.gz
```

Choose an output path that does not already exist. The exporter uses
`config/submission_files.json` to include release files and excludes Git history,
local settings, data and checkpoints. It runs without Git. Optionally pass
`--torch-brain-wheel /path/to/torch_brain-<version>-py3-none-any.whl` to bundle the
exact dependency artifact pinned in `config/torch_brain_dependency.json` and
update the exported installation command. Other dependencies must still be
installed separately.

## Licensing

See [THIRD_PARTY.md](THIRD_PARTY.md) and [LICENSE.txt](LICENSE.txt).
BaRISTA retains its [upstream license](LICENSES/BaRISTA-LICENSE.md).
