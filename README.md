# iMINDBench

iEEG Multi-Insitution Neural Decoding Benchmark codebase. Includes preprocessing, models, and evaluation code. Obtain datasets w/ splits from [torch_brain](https://github.com/neuro-galaxy/torch_brain/tree/gc/add-seeg-movie-watching-datasets) (see [Prepare data](#2-prepare-data)).

[![Paper](https://img.shields.io/badge/arXiv-2609.18104-red)](http://arxiv.org/abs/2609.18104)
[![Website](https://img.shields.io/badge/Website-blue)](https://imindbench.github.io/)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-orange)](https://imindbench.github.io/leaderboard/)
[![Dataset](https://img.shields.io/badge/Dataset-teal)](https://github.com/neuro-galaxy/torch_brain/tree/gc/add-seeg-movie-watching-datasets)

[Getting started](#getting-started) | [Prepare data](#2-prepare-data) | [Full benchmark](#evaluate-the-complete-benchmark) | [Preprocessing](#preprocessing) | [Pretrained weights](#pretrained-weights) | [Customize](#customize) | [Outputs](#outputs) | [Changelog](CHANGELOG.md) | [Citation](#citation)

## Getting started

### 1. Install

CPU evaluation is tested on Python 3.10–3.13; GPU models on newer Python versions
are not yet validated. The default environment uses Python 3.10. From this project directory:

```bash
conda env create -f environment.yml
conda activate imindbench
python -m pip install "torch_brain @ git+https://github.com/neuro-galaxy/torch_brain.git@e39f48ce0ec8c8f59be2507dca8ae172cce79d28"
python -m pip install -e '.[models]'
python -m pip check
```

- Includes the software dependencies for all bundled models.
- The editable installation lets you change code or configs without reinstalling.
- Requires internet access and Git; GPU models also need a compatible CUDA runtime.
- Obtain pretrained weights separately; see [Pretrained weights](#pretrained-weights).

### 2. Prepare data

The first pipeline step is **`brainsets prepare <dataset>`**:

```bash
brainsets prepare neuroprobe_2025 --raw-dir /path/to/raw --processed-dir /path/to/processed
```

- Replace `/path/to/...` with absolute paths outside the checkout.
- Preparation downloads recordings and creates an environment for each pipeline.
- Allow sufficient storage and follow the dataset's terms.

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
BYD and PIPPI configs select their own subdirectories under that root. These two
fields are enough for the first baseline. Add checkpoint/cache paths only when
needed; omitted optional resources inherit packaged `null` defaults.

### 4. Run a first model

Run one CPU Logistic evaluation:

```bash
python -m imindbench.launch --dataset neuroprobev2 --model logistic --experiment default \
  --preprocessor multi_stft_2048Hz --task onset --target sub1_sess1 \
  --device cpu --paths local --config-dir /path/to/config --output-root /path/to/runs/logistic
```

This baseline needs no pretrained weights. Existing output JSONs are skipped.
Remote logging is disabled by default. For full experiments, see below. 

## Evaluate the complete benchmark

We provide scripts to run for each dataset: 
| Script | What it does |
| --- | --- |
| [run_neuroprobev2.sh](scripts/run_neuroprobev2.sh) | NeuroprobeV2: within-session, optional transfer and sample efficiency |
| [run_kelesbyd2024.sh](scripts/run_kelesbyd2024.sh) | BYD: within-session and optional transfer |
| [run_berezutskayapippi2022.sh](scripts/run_berezutskayapippi2022.sh) | PIPPI: within-session and optional transfer |

Open the script for your dataset. Each file lists all benchmark subject/session
pairs and all 15 tasks; the default model is Logistic with multi-STFT inputs.

| Setting | Where to edit it |
| --- | --- |
| Model and inputs | `MODEL` and `PREPROCESSOR` in the script; use its compatibility table |
| Cohort and training-sample caps | `EXPERIMENT=default` or `EXPERIMENT=decodable` in the script |
| Paths and evaluation selection | `CONFIG_DIR`, `OUTPUT_ROOT`, `TASKS` and `TARGETS` in the script |
| Pretrained checkpoints | `popt_checkpoint`, `brainbert_checkpoint`, `barista_checkpoint` or `diver_checkpoint` in `paths/local.yaml` |
| Model architecture and training hyperparameters | `imindbench/conf/model/<MODEL>.yaml` |

After configuring the selected model, run the dataset script:

```bash
bash scripts/run_neuroprobev2.sh
```

**Within-session runs the one pairing you selected** across the listed tasks
and targets. For example, NeuroprobeV2 defaults to:

```bash
MODEL=logistic
PREPROCESSOR=multi_stft_2048Hz
EXPERIMENT=default
```

`EXPERIMENT` selects the cohort and training-sample caps independently of the
model/input pairing:

- `default`: unfiltered cohort, no training-sample cap.
- `decodable`: decodable cohort with automatic training-sample caps.

Training hyperparameters live in the model YAML. Supported inputs are:

| Models | Input preprocessing |
| --- | --- |
| Logistic, MLP, CNN | Multi-STFT or 500 Hz waveforms |
| HTNet | 500 Hz waveforms |
| PopT-v2 (`popt` config) | Multi-STFT matching its checkpoint |
| BrainBERT + linear readout | BrainBERT-specific STFT |
| BaRISTA | Session waveforms with its normalization |
| DIVER-1 | Filtered 500 Hz waveforms with 15-second context |

- **Within-session is enabled by default.** The other experiment commands are commented out.
- Uncomment a whole command to include it; comment out within-session to run only another family.
- Every block uses `MODEL` and `PREPROCESSOR` directly.
- For transfer, select `MODEL=popt` and its matching multi-STFT input.
  Transfer blocks use `TRANSFER_EXPERIMENT=decodable`; within-session and sample
  efficiency use `EXPERIMENT`.

| Family block | Scope |
| --- | --- |
| `within_session` | All selected tasks/targets for the chosen model/input pairing |
| `within_dataset` | PopT-v2; training across sessions within the dataset, Main cohort |
| `multi_dataset` | PopT-v2; training across all three datasets, Main cohort |
| `sample_efficiency` | NeuroprobeV2 only; selected Logistic, MLP, CNN or PopT-v2 on multi-STFT, training fractions 1, 1/2, 1/4, 1/8, 1/16 |

**Complete within-session coverage: 15 tasks × 39 subject/session pairs = 585
evaluations per model/input pairing**, before folds.

- Covers the benchmark selection, not every recording in the original datasets.
- Applies no decodable-target filter.
- Scripts own the task/target selections; coverage tests check that launch previews
  match those explicit lists.

<details>
<summary>Smaller runs and cohort details</summary>

**Smaller runs and settings**

- **Smoke test:** keep at least one entry each in `TASKS` and `TARGETS`.
- **Training:** edit the [model YAML](imindbench/conf/model/) for optimizer,
  stopping criteria, and `epoch_based` or `steps_based` training.
- **Runtime:** edit [runtime/default.yaml](imindbench/conf/runtime/default.yaml)
  for loader workers, thread limits, and determinism.
- **Execution:** enabled script blocks run serially; failures stop the next block.
- **Results:** valid existing JSONs are skipped. Use a new output root after changing settings.

**Dataset subsets**

| Dataset | Default subset |
| --- | --- |
| NeuroprobeV2 | `lite` |
| BYD | `full` |
| PIPPI, including DIVER | `high-cov` |

- Subset tiers choose recordings/splits; `TASKS` and `TARGETS` choose evaluations.
- Listed PIPPI within-session targets use identical splits/channels in `high-cov` and `full`.

**Cohorts and sample caps**

| Preset | Cohort | Per-subject/session training cap |
| --- | --- | --- |
| [default](imindbench/conf/experiment/default.yaml) | Unfiltered | None |
| [decodable](imindbench/conf/experiment/decodable.yaml) | Validation-selected Main cohort | `auto`: target session's training-sample count |

- `default` is selected when no experiment is specified.
- Dataset scripts use `EXPERIMENT` for within-session/sample efficiency and
  `TRANSFER_EXPERIMENT=decodable` for transfer. The sample cap is independent of the model.
- The launcher filters evaluation targets; the data adapter filters training recordings.
  Direct `imindbench.run_eval` calls keep their explicit evaluation target.
- For custom transfer cohorts, add `--decodable-rule NAME` or `--decodable-dir /path/to/manifests`.
- The loader calls within-dataset training `hold-in-session`.

</details>

## Preprocessing

- **Select:** choose a YAML from [imindbench/conf/preprocessor/](imindbench/conf/preprocessor/).
  Use its filename without `.yaml` as script `PREPROCESSOR` or launcher `--preprocessor`.
- **Input rate:** BYD uses 1000 Hz; NeuroprobeV2/PIPPI use 2048 Hz.
- **Structure:** `chain:` lists stages in order. Each stage has a `name:` and parameters;
  the overall chain has no name.
- **Example:** [multi_stft_2048Hz.yaml](imindbench/conf/preprocessor/multi_stft_2048Hz.yaml)
  applies notch filtering → Laplacian referencing → Multi-STFT → training-fitted
  normalization, with **no time-domain high-pass filter**.
- **Customize:** copy a preset into your external config's `preprocessor/` folder,
  edit it, and select its filename. Use a fresh output root. See [Customize](#customize) for new stages.
- **Submit:** `population_*.json` saves settings in `config.preprocess`.
  Those settings determine the leaderboard track; filenames do not.

| Track | Required preprocessing | Bundled configs |
| --- | --- | --- |
| **Multi-STFT** | Standard notch filtering and Laplacian referencing; the benchmark's three STFT resolutions; training-fitted normalization per channel/frequency bin. | `multi_stft_{rate}Hz` |
| **Waveform** | 0.5 Hz high-pass filtering, supported notch filtering, and Laplacian referencing; supported sampling and normalization variants are allowed. | `wav_hpf_robust_*`, `wav_hpf_zscore_*`, `wav_barista_*`, `wav_diver_*` |
| **Custom** | Routes outside the standard track requirements; document the changes and provide a matched baseline where possible. | `stft_*`, `stft_brainbert_*`, `multi_stft_zscore_*`, `wav_nohpf_robust_*` |

- [Submission guide](https://github.com/imindbench/imindbench.github.io/blob/main/leaderboard/README.md#preprocessing-tracks): eligibility and track checks.
- [Preprocessing reference](docs/preprocessing-reference.md): preset recipes and processing rules.
- [Changelog](CHANGELOG.md): behavior changes and upgrade actions.

## Pretrained weights

Store model weights outside the checkout and configure their absolute paths.
Model dependencies are included in the installation above.

| Model | Weights | Configuration |
| --- | --- | --- |
| PopT-v2 | Available upon request. | `paths.popt_checkpoint` |
| BrainBERT | [Official weights ZIP](https://drive.google.com/file/d/14ZBOafR7RJ4A6TsurOXjFVMXiVH6Kd_Q/view?usp=sharing), linked by the [upstream project](https://github.com/czlwang/BrainBERT#using-brainbert-embeddings) | `paths.brainbert_checkpoint` |
| BaRISTA | Available upon request. | `paths.barista_checkpoint` |
| DIVER-1 | [Official iEEG checkpoint](https://drive.google.com/file/d/1svTMyxABZ-9kvk-BiiZ6-2sNyZ5io8mg/view), linked by the [upstream project](https://github.com/DIVER-Project/DIVER-1#weights) | `paths.diver_checkpoint` and writable `paths.diver_shape_cache_dir` |

<details>
<summary>Checkpoint compatibility and DIVER configuration</summary>

| Model | Compatibility notes |
| --- | --- |
| PopT-v2 | Use the multi-STFT weights shared upon request. Accepts `model_cfg`/`model` or `config`/`model_state` checkpoint dictionaries. |
| BrainBERT | Expects the upstream `model_cfg`/`model` format. |
| BaRISTA | Use the weights shared upon request and prepared Destrieux metadata. The `barista` preset selects `localization_Destrieux` for NeuroprobeV2 or `label_destrieux` for BYD/PIPPI. |

For DIVER, select its table entry and add these fields to `paths/local.yaml`:

```yaml
diver_checkpoint: /path/to/ieeg_checkpoint.pt
diver_shape_cache_dir: /path/to/diver_shapes
```

- **Default architecture:** width 256, depth 12, patch size 50; DeepSpeed `module` format.
- **Other checkpoints:** match their architecture. Set `model.deepspeed_pth_format=false`
  for `model_state_dict` format.
- **Validation:** the linked checkpoint passed strict weight loading and a synthetic
  CPU forward pass with fresh and existing shape caches.
- Record the checkpoint hash with your results.

</details>

## Customize

```text
brainsets prepare → prepared recordings and labels → dataset task/split
  → preprocessor chain → model training/evaluation → result JSON and logs
```

| What to change | Location |
| --- | --- |
| Dataset and split settings | `imindbench/conf/dataset/` |
| Model settings / implementation | `imindbench/conf/model/` / `imindbench/models/` |
| Preprocessing settings / implementation | `imindbench/conf/preprocessor/` / `imindbench/preprocessors/` |
| Cohort / runtime presets | `imindbench/conf/experiment/` / `imindbench/conf/runtime/` |
| Task and recording selections | `TASKS` and `TARGETS` in each dataset script |
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
imindbench-grid --dataset neuroprobev2 --model mlp --preprocessor multi_stft_2048Hz --experiment my_trial --task onset --target sub1_sess1 --device cuda:0 --config-dir /path/to/config --paths local --output-root /path/to/runs/my_trial
```

- Keep model settings in YAML for review and reuse.
- Dataset scripts select model/preprocessor/experiment; use `imindbench-grid` for new combinations.

</details>

<details>
<summary>Compare preprocessors using a baseline model</summary>

Select a different preprocessor config while keeping the model, task and unit
selection fixed. This runs Logistic on one NeuroprobeV2 task/recording:

```bash
imindbench-grid --dataset neuroprobev2 --model logistic --preprocessor stft_2048Hz --experiment default --task onset --target sub1_sess1 --device cpu --config-dir /path/to/config --paths local --output-root /path/to/runs/single_stft
```

- Repeat with `--preprocessor multi_stft_2048Hz` and a different output root.
- Use the 1000 Hz config for BYD and the 2048 Hz config for NeuroprobeV2/PIPPI.
- For a custom chain, copy a compatible YAML to
  `/path/to/config/preprocessor/my_chain.yaml`, edit `chain`, and select
  `--preprocessor my_chain`.

</details>

<details>
<summary>Add your own model</summary>

1. Add `imindbench/models/my_model.py` and register it with `@register_model("my_model")`.
   Modules in this directory are discovered automatically.
2. Implement the interface for your model type:

| Type | Starting point | Interface |
| --- | --- | --- |
| PyTorch | `mlp_model.py` | Subclass `TorchBaseModel`; implement `_create_network(input_shape, n_classes)` and `build_model(input_shape, n_classes, device=None)`. The runner owns training. |
| sklearn | `logistic_model.py` | Implement `fit` and `predict_proba`, set `classes_`, and use `prepare_batch` if input adaptation is needed. Probabilities must follow `classes_` order. |

3. Add `imindbench/conf/model/my_model.yaml` with `name: my_model`, required
   `backend: torch` or `backend: sklearn`, input requirements and training settings.
   The backend selects the runner; `--device` applies only to Torch models.
4. Select `--model my_model` with the dataset, preprocessor and path arguments
   from the custom experiment example.

- Start with one task/target; check input shapes and class probabilities.
- Expand `TASKS` and `TARGETS`, then repeat across all three datasets with matching preprocessors.
- Specify both task and target selections explicitly.

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

- Add `- {name: gain, factor: 2.0}` to a copied preprocessor chain.
- Preserve sample metadata; keep shapes, channel labels and coordinates aligned.
- For transforms that learn statistics, use `execution_type = "fold_fit_transform"`
  and implement `fit_split`, `get_state`, `set_state` and `reset_state`.
  Fit on training data only.
- See `standardization_preprocessor.py` for a stateful example.

</details>

## Outputs

Each run writes `population_*.json`, resolved Hydra config, `launch.json` and
`launcher.log`.

| Situation | Behavior |
| --- | --- |
| Valid output JSON exists | Skip the evaluation |
| Output JSON is missing or malformed | Restart the evaluation; no checkpoint resume |
| Settings change | Use a new output root; existing results are reused by filename |
| Concurrent launchers | Use separate output roots |

The launcher accepts `--dry-run` to preview commands or `--count` to count evaluations.

<details>
<summary>Development checks</summary>

```bash
python -m pip install -e '.[models,dev]'
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
```

Tests use synthetic data; optional model checks skip when their dependencies are unavailable.

</details>

See [LICENSE.txt](LICENSE.txt) and [third-party notices](THIRD_PARTY.md).
BaRISTA retains its [upstream license](LICENSES/BaRISTA-LICENSE.md).
PopT retains its [upstream MIT license](LICENSES/PopT-LICENSE.txt), with separate
terms for tutorial, PyTorch, and SciPy portions described in the third-party notices.
DIVER-1 retains its [upstream MIT license](LICENSES/DIVER-1-LICENSE.txt), with
separate Apache-2.0 terms for uni2ts portions described in the third-party notices.

## Citation

```bibtex
@misc{chau2026imindbench,
  title={{iMINDBench}: {iEEG} Multi-Institution Neural Decoding Benchmark},
  author={Geeling Chau and Saba Hashemi and Yonghyeon Gwon and Eshani Patel and Jan DeWitt and Christopher Wang and Andrii Zahorodnii and Sabera J Talukder and Danny Dongyeop Han and Chun Kee Chung and Maryam M Shanechi and Yisong Yue},
  year={2026},
  eprint={2609.18104},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={http://arxiv.org/abs/2609.18104},
}
```
