# iMINDBench

An independently installable benchmark for intracranial neural decoding.
TorchBrain provides public data preparation and dataset loaders; iMINDBench
provides preprocessing, models, and evaluation.

Start with the [installation and single-recording quickstart](imindbench/README.md).
For larger runs, see the [three examples and recipe catalog](docs/EXPERIMENTS.md).
Required checkpoints are listed in the [resource guide](docs/RESOURCES.md).
For reviewer distribution, see [submission packaging](docs/SUBMISSION.md).

The retained model families are Logistic, Linear/BrainBERT, MLP, CNN, HTNet,
PopT, DIVER, and BaRISTA. Checkpoint-backed models require separate resources.
The package does not bundle recordings or pretrained weights.

Existing historical results and parity evidence are preserved. See the
[reproduction coverage](docs/REPRODUCTION.md) for what has been verified and
which historical cases remain limited by slicing or checkpoint provenance.

Development checks use the same Ruff rule families as TorchBrain:

```bash
python -m pip install -e '.[dev]'
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Install the documented public TorchBrain dependency first. An older editable
TorchBrain checkout may expose a different API even under the same package name.
