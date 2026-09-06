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

This setup supports Logistic, MLP, CNN and HTNet without pretrained weights.
TorchBrain provides data preparation and loaders; install it at the version
shown above, without the former standalone BrainSets package. Optional model
dependencies are listed in [Experiments](docs/EXPERIMENTS.md#optional-models).

## Prepare public data

For the first example, prepare Neuroprobe data outside the checkout:

```bash
brainsets prepare neuroprobe_2025 --raw-dir /path/to/raw --processed-dir /path/to/processed
```

Preparation may download a large dataset and creates an isolated preparation
environment. NeuroprobeV2 uses these same prepared artifacts. Preparation
commands for BYD and PIPPI are in [Experiments](docs/EXPERIMENTS.md#other-datasets).

Create an external paths file from the example:

```bash
mkdir -p /path/to/config/paths
cp imindbench/conf/paths/example.yaml /path/to/config/paths/local.yaml
```

Edit `paths/local.yaml`: set `dataset_root` to your processed-data root and
`dataset_dirname` to its dataset directory. Leave checkpoint paths null for the
first example. Replace the `/path/to/...` placeholders with your own absolute paths.

## First evaluation

From the repository or extracted source archive, preview one CPU Logistic evaluation:

```bash
bash scripts/run_logistic.sh \
  --config-dir /path/to/config --output-root /path/to/runs/logistic
```

Then run it:

```bash
bash scripts/run_logistic.sh \
  --config-dir /path/to/config --output-root /path/to/runs/logistic --execute
```

The scripts use the active Python environment. Outputs include `population_*.json`,
the resolved configuration, launch command and log. Add `--resume` to an executed
command to reuse verified completed outputs. Remote logging is disabled by default.

Use `scripts/run_mlp.sh` for a GPU example or `scripts/run_barista.sh` after
installing its optional dependencies and setting a checkpoint. Each script selects
one NeuroprobeV2 onset evaluation; see [Experiments](docs/EXPERIMENTS.md) for
customization and larger grids. The same scripts are exercised by the test suite
in dry-run mode, without downloading data or training models.

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
