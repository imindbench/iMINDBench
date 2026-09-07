# iMINDBench

Intracranial neural decoding across three movie-watching datasets. TorchBrain
prepares recordings and supplies dataset loaders; iMINDBench provides
preprocessing, models and evaluation.

## Getting started

### 1. Install

Use Python 3.10. From this project directory:

```bash
conda env create -f environment.yml
conda activate imindbench
python -m pip install "torch_brain @ git+https://github.com/neuro-galaxy/torch_brain.git@e39f48ce0ec8c8f59be2507dca8ae172cce79d28"
python -m pip install -e '.[models]'
python -m pip check
```

This installs the software dependencies for the included models. Keep this
editable checkout to change code or configs without reinstalling. Installation
requires internet access and Git; GPU models need a compatible CUDA runtime.
Pretrained weights are downloaded separately using the table below.

### 2. Prepare data

The first pipeline step is **`brainsets prepare <dataset>`**:

```bash
brainsets prepare neuroprobe_2025 --raw-dir /path/to/raw --processed-dir /path/to/processed
```

Replace `/path/to/...` with your own absolute paths outside the checkout.
Preparation downloads recordings and builds a separate environment for each
pipeline's dependencies. Allow sufficient storage and follow the dataset's terms.

| Dataset | `brainsets prepare` name | Evaluation name | Benchmark subject/session pairs |
| --- | --- | --- | ---: |
| NeuroprobeV2 | `neuroprobe_2025` | `neuroprobev2` | 5 |
| Bang! You're Dead | `keles_byd_2024` | `kelesbyd2024` | 29 |
| PIPPI | `berezutskaya_pippi_2022` | `berezutskayapippi2022` | 5 |

Prepare each dataset into the same processed root to run the complete benchmark.
Each pipeline creates its own named subdirectory. Neuroprobe2025 and NeuroprobeV2
use the same prepared artifacts through different dataset views.

### 3. Set paths

```bash
mkdir -p /path/to/config/paths
cp imindbench/conf/paths/example.yaml /path/to/config/paths/local.yaml
```

Set `dataset_root: /path/to/processed` and `dataset_dirname: neuroprobe_2025`.
BYD and PIPPI configs select their own subdirectories under that root. Keep every
key; checkpoint/cache fields can remain `null` for the first baseline.

### 4. Run a first model

Run one CPU Logistic evaluation:

```bash
python -m imindbench.launch --dataset neuroprobev2 --model logistic --experiment multi_stft/logistic \
  --preprocessor laplacian_multi_stft_2048Hz --task onset --target sub1_sess1 \
  --device cpu --paths local --config-dir /path/to/config --output-root /path/to/runs/logistic
```

This baseline needs no pretrained weights. Existing output JSONs are skipped.
Remote logging is disabled by default. For full experiments, edit a dataset script:

| Script | What it does |
| --- | --- |
| [run_neuroprobev2.sh](scripts/run_neuroprobev2.sh) | NeuroprobeV2: within-session, optional transfer and sample efficiency |
| [run_kelesbyd2024.sh](scripts/run_kelesbyd2024.sh) | BYD: within-session and optional transfer |
| [run_berezutskayapippi2022.sh](scripts/run_berezutskayapippi2022.sh) | PIPPI: within-session and optional transfer |

## Evaluate the complete benchmark

Open the script for your dataset and choose `MODEL`, `PREPROCESSOR` and
`EXPERIMENT` from its commented compatibility table. The default is Logistic
with multi-STFT inputs. Edit `CONFIG_DIR`, `OUTPUT_ROOT`, `TASKS` and `TARGETS`;
each file lists all benchmark subject/session pairs and all 15 tasks.
Configure pretrained checkpoints and model settings in YAML before launching.
Set `popt_checkpoint`, `brainbert_checkpoint`, or `barista_checkpoint` in
`paths/local.yaml`. Edit model settings in `imindbench/conf/model/<MODEL>.yaml`
and training overrides in the selected `imindbench/conf/experiment/<EXPERIMENT>.yaml`;
experiment values take precedence over model values. BrainBERT automatically
uses the model device.

```bash
bash scripts/run_neuroprobev2.sh
```

**Within-session runs the one pairing you selected** across the listed tasks
and targets. For example, NeuroprobeV2 defaults to:

```bash
MODEL=logistic
PREPROCESSOR=laplacian_multi_stft_2048Hz
EXPERIMENT=multi_stft/logistic
```

`EXPERIMENT` selects training defaults; change it along with the pairing using
the table in the script. Supported inputs are:

| Models | Input preprocessing |
| --- | --- |
| Logistic, MLP, CNN | Multi-STFT or 500 Hz waveforms |
| HTNet | 500 Hz waveforms |
| PopT-v2 (`popt` config) | Multi-STFT matching its checkpoint |
| BrainBERT + linear readout | BrainBERT-specific STFT |
| BaRISTA | Session waveforms with its normalization |
| DIVER-1 | Filtered 500 Hz waveforms with 15-second context |

Each experiment command includes the shared constants directly; there is no
`launch` wrapper. The within-session command runs below the settings. Within-dataset
and multi-dataset commands are commented out below it. Uncomment a whole command
to include it, and comment out the within-session command to run only another
family. Every block uses `MODEL` and `PREPROCESSOR`. For within-dataset or
multi-dataset, select `MODEL=popt` and its matching multi-STFT preprocessor;
these blocks use `TRANSFER_EXPERIMENT=decodable` for the Main cohort and sample
cap. Within-session and sample efficiency use `EXPERIMENT`. NeuroprobeV2 also
includes a commented sample-efficiency command.

| Family block | Scope |
| --- | --- |
| `within_session` | All selected tasks/targets for the chosen model/input pairing |
| `within_dataset` | PopT-v2; training across sessions within the dataset, Main cohort |
| `multi_dataset` | PopT-v2; training across all three datasets, Main cohort |
| `sample_efficiency` | NeuroprobeV2 only; selected Logistic, MLP, CNN or PopT-v2 on multi-STFT, training fractions 1, 1/2, 1/4, 1/8, 1/16 |

Within-session covers 15 tasks × 39 benchmark subject/session pairs across the
three scripts: **585 evaluations per model/input pairing**, before folds.
This is the benchmark selection, not every recording in the original datasets.
No decodable-target filter is applied. Coverage tests compare the scripts' lists
with [units/catalog.yaml](imindbench/conf/units/catalog.yaml).

<details>
<summary>Smaller runs and cohort details</summary>

For a smoke test, leave only the desired entries in `TASKS` and `TARGETS`.
The scripts execute directly and skip evaluations whose output JSON exists.
Keep at least one task and subject/session pair. Edit training settings such as
`max_iter` in the selected model or experiment YAML. Keep checkpoint paths in
`paths/local.yaml`, except DIVER's `upstream_ckpt` and `model_dir`, which belong
in `imindbench/conf/model/diver.yaml`.

Waveform baselines use 15-second context for filtering, then crop to the target
window, apply Laplacian referencing, downsample to 500 Hz and fit robust scaling
on training data. Multi-STFT uses each dataset's native sampling rate.

The scripts inherit dataset defaults: NeuroprobeV2 `lite`, BYD `full`, and PIPPI
`high-cov`, including DIVER. Subset tiers select eligible recordings and prepared
splits; the task/target lists select the evaluation grid. For the listed PIPPI
within-session targets, `high-cov` and `full` use identical splits and channels.
Within-dataset and multi-dataset retain the validation-selected Main
cohort for both targets and training. Dataset YAMLs default `train_decodable_subject_sessions_only` to `false`.
The transfer blocks select the `decodable` preset, which sets it to `true`,
supplies the standard manifest, and filters the script's target list per task. To use a
custom cohort, add `--decodable-rule NAME` or `--decodable-dir /path/to/manifests`
to that command. Within-dataset and multi-dataset also default to
`dataset.max_train_samples_per_subject=auto`: each training subject/session is
capped at the target session's training-sample count. Within-session and
sample-efficiency presets leave this cap disabled; it is not a PopT-v2
requirement. The loader calls within-dataset training `hold-in-session`.

The scripts use the selected constants and the existing Python grid launcher
for task/target expansion, execution and resume. Each enabled block runs serially;
if it reports failures, the script stops before starting the next block. Use a new output root after changing settings.

</details>

## Pretrained weights

Download weights outside the checkout and configure their absolute paths.
The install step above already includes the model dependencies.

| Model | Weights | Configuration |
| --- | --- | --- |
| PopT-v2 | Public multi-STFT checkpoint link pending verification. The [original PopT release](https://huggingface.co/PopulationTransformer/popt_brainbert_stft) uses BrainBERT features and is **not a verified substitute**. | `paths.popt_checkpoint` |
| BrainBERT | [Official weights ZIP](https://drive.google.com/file/d/14ZBOafR7RJ4A6TsurOXjFVMXiVH6Kd_Q/view?usp=sharing), linked by the [upstream project](https://github.com/czlwang/BrainBERT#using-brainbert-embeddings); extract `stft_large_pretrained.pth` | `paths.brainbert_checkpoint` |
| BaRISTA | [Official checkpoints](https://huggingface.co/ShanechiLab/BaRISTA/tree/main/pretrained_models); select `parcels_chans.ckpt` for the included Destrieux configuration | `paths.barista_checkpoint` |
| DIVER-1 | [Official iEEG checkpoint](https://drive.google.com/file/d/1svTMyxABZ-9kvk-BiiZ6-2sNyZ5io8mg/view), linked by the [upstream project](https://github.com/DIVER-Project/DIVER-1#weights) | `model.upstream_ckpt` and writable `model.model_dir` |

<details>
<summary>Checkpoint compatibility and DIVER configuration</summary>

The PopT-v2 download remains a release gap; use your compatible multi-STFT
checkpoint until that link is available. PopT accepts `model_cfg`/`model` or
`config`/`model_state` checkpoint dictionaries. BrainBERT expects the upstream
`model_cfg`/`model` format. BaRISTA uses parcel embeddings and requires the
prepared Destrieux metadata. The `barista` experiment preset automatically selects
`localization_Destrieux` for NeuroprobeV2 or `label_destrieux` for BYD/PIPPI.

For DIVER, select its table entry and edit these fields in
`imindbench/conf/model/diver.yaml`:

```yaml
upstream_ckpt: /path/to/ieeg_checkpoint.pt
model_dir: /path/to/diver_shapes
```

The linked DIVER iEEG checkpoint matches the defaults: width 256, depth 12,
patch size 50 and DeepSpeed `module` format. It was checked with strict weight
loading and a synthetic CPU forward pass using fresh and existing shape caches.
For other checkpoints, match their architecture and set
`model.deepspeed_pth_format=false` if they use `model_state_dict` format.
Record the checkpoint hash with your results.

</details>

## Customize the pipeline

```text
brainsets prepare → prepared recordings and labels → dataset task/split
  → preprocessor chain → model training/evaluation → result JSON and logs
```

| What to change | Location |
| --- | --- |
| Dataset and split settings | `imindbench/conf/dataset/` |
| Model settings / implementation | `imindbench/conf/model/` / `imindbench/models/` |
| Preprocessing settings / implementation | `imindbench/conf/preprocessor/` / `imindbench/preprocessors/` |
| Training / runtime presets | `imindbench/conf/experiment/` / `imindbench/conf/runtime/` |
| Task and unit catalog | `imindbench/conf/units/catalog.yaml` |
| Launching / single evaluation | `imindbench/launch.py` / `imindbench/run_eval.py` |

Preparation pipelines and dataset loader implementations live in TorchBrain,
under `torch_brain/pipeline/brainsets-pipelines/` and `torch_brain/datasets/`.

<details>
<summary>Change settings with a small YAML file</summary>

Create `/path/to/config/experiment/my_trial.yaml`:

```yaml
# @package _global_
model:
  max_iter: 20
  learning_rate: 0.001
```

Select it through the generic launcher:

```bash
imindbench-grid --dataset neuroprobev2 --model mlp --preprocessor laplacian_multi_stft_2048Hz --experiment my_trial --task onset --target sub1_sess1 --device cuda:0 --config-dir /path/to/config --paths local --output-root /path/to/runs/my_trial
```

Keep model settings in the YAML file so the configuration is easy to review and
reuse. Dataset scripts select the model, preprocessor and experiment; use
`imindbench-grid` for new combinations.

</details>

<details>
<summary>Compare preprocessors using a baseline model</summary>

Select a different preprocessor config while keeping the model, task and unit
selection fixed. This runs Logistic on all NeuroprobeV2 units/tasks:

```bash
imindbench-grid --dataset neuroprobev2 --model logistic --preprocessor laplacian_stft_2048Hz --experiment baseline --unit-set all --device cpu --config-dir /path/to/config --paths local --output-root /path/to/runs/single_stft
```

Repeat with `--preprocessor laplacian_multi_stft_2048Hz` and a different output
root. Select the 1000 Hz config for
BYD and the 2048 Hz config for NeuroprobeV2/PIPPI. To create a custom chain, copy
a compatible YAML to `/path/to/config/preprocessor/my_chain.yaml`, edit its
`chain` entries and select `--preprocessor my_chain`.

</details>

<details>
<summary>Add your own model</summary>

1. Add `imindbench/models/my_model.py`. Start from `mlp_model.py` for PyTorch or
   `logistic_model.py` for sklearn, and register it with `@register_model("my_model")`.
2. For PyTorch, subclass `TorchBaseModel` and implement
   `_create_network(input_shape, n_classes)` and
   `build_model(input_shape, n_classes, device=None)`. The runner owns training.
   For sklearn, implement `fit` and `predict_proba`, set `classes_`, and use
   `prepare_batch` to adapt inputs if necessary. Probabilities must follow `classes_` order.
3. Add `imindbench/conf/model/my_model.yaml` with `name: my_model`, input requirements
   and training settings. Select it with `imindbench-grid --model my_model` and the
   dataset/preprocessor/path arguments from the custom experiment example.

Modules in `imindbench/models/` are discovered automatically. Start with one
unit/task, check the input shape and class probabilities, then omit `--task`
and `--target` to evaluate the full catalog for that dataset. Repeat for all
three datasets with matching preprocessors to evaluate a custom model completely.

</details>

<details>
<summary>Add your own preprocessor</summary>

Add a module under `imindbench/preprocessors/` and register its class. For example:

```python
from imindbench.preprocessors import register_preprocessor
from imindbench.preprocessors.base_preprocessor import BasePreprocessor

@register_preprocessor("gain")
class GainPreprocessor(BasePreprocessor):
    def transform_samples(self, samples):
        return [{**sample, "x": sample["x"] * self.cfg.factor} for sample in samples]
```

Add `- {name: gain, factor: 2.0}` to a copied preprocessor chain. Preserve sample
metadata and keep shapes, channel labels and coordinates aligned. If a transform
learns statistics, use `execution_type = "fold_fit_transform"` and implement
`fit_split`, `get_state`, `set_state` and `reset_state`; fit on training data only.
See `standardization_preprocessor.py` for a stateful example.

</details>

## Outputs and resume

Each run writes `population_*.json`, resolved Hydra config, `launch.json` and
`launcher.log`. Use a separate output root for each configuration or independent
worker; one launcher locks its root while executing.

Dataset scripts skip an evaluation whenever its output JSON already exists,
matching the original experiment scripts. No saved-command match or checksum
is required. Evaluations without a result JSON run from the beginning; this does
not restore training checkpoints. Use a new output root when changing settings,
since an existing result is reused based on its filename alone.

Execution and skipping existing results are also the defaults for `imindbench-grid`.
Use `--dry-run` to print commands without running, or `--count` to count the grid.

<details>
<summary>Development checks</summary>

```bash
python -m pip install -e '.[models,dev]'
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Tests cover configurations, full-catalog launch previews and synthetic runtime
contracts without downloading complete datasets or rerunning benchmark experiments.
Some tests need external checkpoints and skip when those are unavailable.

</details>

See [LICENSE.txt](LICENSE.txt) and [third-party notices](THIRD_PARTY.md).
BaRISTA retains its [upstream license](LICENSES/BaRISTA-LICENSE.md).
