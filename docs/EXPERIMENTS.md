# Experiment launchers

Use `imindbench` for a single evaluation. The retained shell scripts describe
larger historical experiment grids; inspect their model/task/regime arrays
before launching. They do not require Git or write into their installation.
Their existence does not establish exact historical reproduction for every grid.

Set `IMINDBENCH_OUTPUT_ROOT` to an absolute, caller-owned output directory.
Results, Hydra logs and multisource locks live beneath it. Scripts preserve their
existing result layouts, skip existing result JSONs and report failed evaluations.
Use a fresh output root when changing a configuration: file-existence skipping
does not validate the provenance of an existing result.

For machine settings outside the installation, create a directory with a
`paths/local.yaml` file copied from `imindbench/conf/paths/example.yaml`, then set
`IMINDBENCH_CONFIG_DIR` to that directory and `PATHS_CFG=local`.
Use absolute data/checkpoint/cache paths. Optional `TMPDIR` selects a short
absolute temporary directory; the default is `/tmp`. `BARISTA_TMPDIR` remains
accepted for existing BaRISTA workflows.

```bash
IMINDBENCH_OUTPUT_ROOT=/path/to/runs \
IMINDBENCH_CONFIG_DIR=/path/to/config \
PATHS_CFG=local \
bash /path/to/imindbench/scripts/run_neuroprobev2.sh
```

The script path can be inside an installed package or a source archive. Positional
arguments are forwarded to Hydra. Use `--cfg job` to inspect generated configs
without training; a grid still prints one config per evaluation. Explicit
overrides should not change the loop's model/task/subject identity, since those
fields determine output paths and prechecks.

| Goal | Existing script family |
| --- | --- |
| Baselines across tasks/regimes | `run_neuroprobev2.sh`, `run_byd.sh`, `run_pippi.sh` |
| STFT variations | `run_*_stft_sweep.sh` |
| BrainBERT preprocessing/model | `run_brainbert_*.sh` |
| BaRISTA with supplied checkpoint | `run_*_barista.sh` |
| Decodable hold-in/multisource | `run_*_decodable_subjects_*.sh` |

The decodability-rule wrappers select a packaged population manifest. Multisource
wrappers also use `flock` and a startup delay controlled by
`MULTISOURCE_START_DELAY_SECONDS` (default 240 seconds). For shell launchers,
provide `DECODABLE_SUBJECT_SESSIONS_DIR` explicitly if using a custom population.

The single-recording quickstart is the introductory example. These grids retain
their scientific selections during the portability refactor; consolidation into
declarative recipes can follow without changing their numerical defaults.
