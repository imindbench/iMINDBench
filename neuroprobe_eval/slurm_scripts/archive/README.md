# Archived SLURM Scripts

Most scripts in this directory were used for older neuroprobe evaluation batch submissions and cluster runs.

This directory is kept mainly for reference and reproducibility. In general, these SLURM scripts should not be treated as the canonical entrypoints for the current pipeline.

Several scripts here were also used for earlier baselines, debug runs, or prior evaluation setups, including:

- archived batch submissions for logistic, MLP, CNN, linear, PopT, and related variants
- debug submission scripts used to validate scheduler setup or partial pipeline behavior
- older resource and layout conventions that may no longer match the current recommended scripts

If one of these scripts needs to be reused, verify that its assumptions still match the current dataset layout, Hydra configs, output paths, and cluster resource expectations before running it.
