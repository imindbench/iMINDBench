# Neuroprobe Eval

Public release surface for running Hydra-based evaluation on supported public providers.

## Scope

Public code path supports:
- dataset provider: `neuroprobe2025`
- dataset provider: `kelesbyd2024`
- dataset provider: `berezutskayapippi2022`
- regimes: `SS-SM`, `SS-DM`, `DS-DM`
- regimes: `within-session`, `hold-in-session`, `hold-out-session`, `hold-out-subject`
- models: `logistic`, `mlp`, `cnn`, `popt`, `htnet`, `brant`, `barista`

HTNet requires the time-domain `laplacian_wav_*` preprocessors and is currently
intended for aligned within-session runs. The shipped HTNet configs are adapted
for the 1 s neuroprobe_eval windows; the Peterson et al. paper tuned HTNet on
2 s segments resampled to 250 Hz, so these defaults should be treated as
task-specific adaptations rather than exact paper-parity hyperparameters.

BRANT uses the BaRISTA-style compatibility path for short windows: each whole
raw window is resampled to one `1500`-sample pseudo-patch before entering the
BRANT time/channel encoders. This is not paper-parity BRANT, which expects
`15 x 6s` patches over `90s` context. The default BRANT classification head is
`mlp_token`, which applies the original-style MLP to each valid channel token
and then averages logits to match Neuroprobe's sample-level labels. Set
`model.head_type=linear_pooled` to reproduce the older pooled-linear local head.

BaRISTA (`model=barista`) is a port of
[ShanechiLab/BaRISTA](https://github.com/ShanechiLab/BaRISTA). It fine-tunes a
pretrained region-tokenizer + transformer upstream, so it needs two things that
have no usable default:

- `paths.barista_checkpoint=<path/to/*.ckpt>` — the pretrained upstream weights.
  Without it the model raises at construction (or set `model.random_init=true`
  for an untrained control).
- `dataset.brain_area_key=<key holding Destrieux labels>` — the spatial encoder
  maps Destrieux region names to embedding ids, and the dataset defaults do not
  point at a Destrieux key. Pass `localization_Destrieux` for neuroprobev2,
  `label_destrieux` for pippi and BYD-destrieux, or `location` for BYD.
  BaRISTA raises if fewer than 50% of a recording's channels resolve to a
  Destrieux region rather than training on a collapsed spatial encoding — on
  neuroprobev2 sub1_sess1, `localization_Destrieux` resolves 120/120 channels
  while the `localization_DesikanKilliany` default resolves only 7/120. Tune
  with `model.min_resolved_region_fraction`, or set
  `model.tokenizer.add_spatial_encoding=false` to opt out deliberately.

BaRISTA also requires `xformers` (see `environment.yml`).

The dataset launchers default to smoke subsets; the complete cell lists are in
`leaderboard/data/coverage_contract.json`. Keep their worker and schedule defaults
when reproducing published results because those settings affect RNG order.

Note on the LR schedule when comparing against BaRISTA numbers recorded before
this port: `epoch_based` training overwrites `model.scheduler.total_steps` with
`max_iter * steps_per_epoch`, so the 5% warm-up resolves to a few epochs. The
older runs used `conf/model/barista.yaml`'s literal `total_steps: 10000`, which
made warm-up longer than the whole run. To reproduce those runs exactly, pin the
schedule in optimizer updates — these take precedence over the override:

```
++model.scheduler.warmup_steps=500 ++model.scheduler.step_size_updates=95
```

## Layout

- `run_eval.py`: main entrypoint
- `scripts/`: simple public regime runners
- `models/`: public model implementations
- `preprocessors/`: public preprocessors
- `conf/`: public Hydra configs

## Environment Setup

Use the conda environment file as the canonical setup:

Before creating or updating the environment, edit
`examples/neuroprobe_eval/environment.yml` and replace:
- `/path/to/brainsets`
- `/path/to/torch_brain`
with your local checkout paths.

```bash
cd torch_brain
conda env create -f examples/neuroprobe_eval/environment.yml
```

If the environment already exists, update it in place:

```bash
cd torch_brain
conda env update -n neuroprobe_eval -f examples/neuroprobe_eval/environment.yml --prune
```

Activate it before running:

```bash
conda activate neuroprobe_eval
```

Optional extras (not required for default runs):
- `wandb`
- `warmup_scheduler`

## Quick Start

From the examples parent directory:

```bash
cd torch_brain/examples
python -m neuroprobe_eval.run_eval --help
```

### Run by regime (recommended)

```bash
./neuroprobe_eval/scripts/run_eval.sh
```

Edit constants at the top of `scripts/run_eval.sh`:

```bash
REGIME
PATHS_CFG
TEST_SUBJECT
TEST_SESSION
TASK
LABEL_MODE
MODEL
PREPROCESSOR
OUTPUT_GROUP
```

Minimal PIPPI PopT runner:

```bash
./neuroprobe_eval/scripts/run_pippi_within_session_popt.sh
```

PIPPI all-regime runners:

```bash
./neuroprobe_eval/scripts/run_pippi.sh
```

`run_pippi.sh` now sweeps `logistic`, `mlp`, `cnn`, and `popt`, and includes a
commented `htnet_2048Hz` option. If you enable HTNet there, switch
`PREPROCESSOR` from `laplacian_stft_2048Hz` to `laplacian_wav_2048Hz`.

PIPPI sampling-rate policy (phase 1):
- `high-cov` runs use native `2048 Hz` preprocessors.
- `low-cov` runs use native `512 Hz` preprocessors.
- `full` mixed-rate runs are not launched directly; aggregate `high-cov` and
  `low-cov` outputs post hoc.
- Folds with fewer than 2 classes in required split(s) are skipped and logged
  with class-count diagnostics (instead of failing mid-run).

### Task-Decodable Subject Filtering

BYD, PIPPI, and NeuroProbeV2 launchers can restrict target runs and eligible
train regimes to subject/session pairs listed as decodable for each task. Set
`DECODABLE_TRAIN_SUBJECT_SESSIONS_ONLY="true"` in the launcher to enable the
filter; dedicated `run_*_decodable_subjects*.sh` scripts set it to `"true"` by
default.

The launchers skip target subject/session runs that are not listed for the task.
Set `decodable_subject_sessions_dir` in the selected `PATHS_CFG`; the launchers
also pass:

```bash
dataset.train_decodable_subject_sessions_only=true
```

Manifest lookup is provider-specific and expects `<provider>.json`. Current
checked-in manifests cover `kelesbyd2024`, `berezutskayapippi2022`, and
`neuroprobev2`.

Scripts with `_multisource` in the filename use `dataset.train_sources`; hold-in
and same-subject variants use the single target dataset config. Launchers that
cap per-subject train samples expose that value as `MAX_TRAIN_SAMPLES_PER_SUBJECT`
and pass it to `dataset.max_train_samples_per_subject`.

When `dataset.train_sources` is set, decodable train filtering supports only
`train_sources[].regime: hold-in-session`. This defines each source's training
set as all eligible task-decodable subject/sessions for that provider, independent
of the eval target subject/session.

## Environment Overrides for Scripts

Scripts are constant-driven; adjust values directly in the files.

## Standardization Modes

The `standardize` preprocessor supports multiple train-fit normalization modes:

- `global_feature`: one mean/std per feature or time index, pooled over training
  samples and channels. For STFT-shaped `(channels, timebins, freqs)` inputs,
  each `(timebin, freq)` position keeps separate statistics.
- `global_scalar`: one mean/std scalar pooled over every training value,
  flattening channels, time/timebins, and feature axes. Intended mainly for
  waveform inputs; for STFT this mixes frequency bins with different natural
  scales.
- `global_robust_scalar`: one approximate robust scalar estimated from a
  bounded reservoir of training values, using median and `1.4826 * MAD`.
  Intended mainly for waveform inputs; for STFT/multi-STFT use as an ablation
  rather than a default.

Recommended use:
- For waveform inputs shaped `(channels, time)`, `global_scalar` and
  `global_robust_scalar` can normalize recording scale while preserving
  cross-channel amplitude relationships.
- For STFT/multi-STFT inputs shaped `(channels, timebins, freqs)`, prefer
  frequency-aware modes such as `global_feature` or
  `per_channel_samples_time_pooled`. A single global scalar is usually only
  appropriate as an ablation because frequency bins can have very different
  natural scales.

`global_robust_scalar` estimates median/MAD from a bounded deterministic
reservoir of training values to avoid materializing every value in memory. The
default `robust_reservoir_size` is `1000000`; smaller values reduce memory use
but make the robust scalar an approximation.

## Direct Hydra Example

```bash
python -m neuroprobe_eval.run_eval \
  paths=<paths_cfg> \
  dataset.provider=neuroprobe2025 \
  dataset.regime=SS-SM \
  dataset.task=onset \
  dataset.test_subject=2 \
  dataset.test_session=0 \
  model=logistic \
  preprocessor=laplacian_stft \
  wandb.enabled=false
```

BYD example:

```bash
python -m neuroprobe_eval.run_eval \
  paths=server_ml7_popt_stft \
  dataset=kelesbyd2024 \
  dataset.test_subject=41 \
  dataset.test_session=1 \
  dataset.task=speech \
  model=popt \
  preprocessor=laplacian_stft_1000Hz \
  wandb.enabled=false
```

BYD HTNet example:

```bash
python -m neuroprobe_eval.run_eval \
  paths=server_ml7_popt_stft \
  dataset=kelesbyd2024 \
  dataset.regime=within-session \
  dataset.test_subject=42 \
  dataset.test_session=1 \
  dataset.task=speech \
  model=htnet_1000Hz \
  preprocessor=laplacian_wav_1000Hz \
  wandb.enabled=false
```

PIPPI example:

```bash
python -m neuroprobe_eval.run_eval \
  paths=processed_1_0_0_pippi \
  dataset=berezutskayapippi2022 \
  dataset.subset_tier=high-cov \
  dataset.regime=within-session \
  dataset.task=speech \
  dataset.test_subject=1 \
  dataset.test_session=1 \
  model=popt \
  preprocessor=popt_stft_filter_neuroprobe_pippi_highcov \
  wandb.enabled=false
```

PIPPI channel coordinates are read in ACPC and transformed to PopT checkpoint
slot order `[L, I, P]` inside the variable-channel adapter. Samples and batches
expose the neutral key `channel_coords`, and the active frame selection is
controlled by `dataset.coordinate_profile` (`popt_lip` by default,
`diver_mni` for the alternate source-frame view).

For aligned multi-subject PIPPI runs (`logistic/mlp/cnn` with non-`within-session`
regimes), use `dataset.brain_area_key=group` and a preprocessor chain that
includes `region_intersection_pool`.
