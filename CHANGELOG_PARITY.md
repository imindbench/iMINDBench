# iMINDBench migration and parity change log

This file is added after the mechanical Neuroprobe Eval baseline commit and
records every subsequent migration change that affects runtime behavior,
configuration, result discovery, or parity. Update it in the same commit as the
corresponding change.

## Source baseline

- Runtime source repository: `git@github.com:glchau/torch_brain.git`
- Runtime source checkout: `/home/geeling/Projects/ieeg_project/torch_brain-main`
- Runtime source branch: `main`
- Runtime source commit: `726d96978c3e573c7a31e2e63b18c905e4c15d6b` (BaRISTA release; later MVPFormer/SEEGnificant releases excluded)
- Notebook/YAML analysis checkout: `/home/geeling/Projects/ieeg_project/torch_brain`
- Notebook/YAML base commit: `47a3c8900c1f6bf2aec79bb08d99c389e4036603` (dirty working tree; per-artifact hashes required)
- Original-result capture root: `/home/geeling/Projects/ieeg_project/torch_brain/examples/neuroprobe_eval/outputs/09_neurips` (read-only)
- Baseline import commit: `441aad6` (`chore: import neuroprobe_eval baseline from 726d969`)
- Baseline projection: 221 files from `examples/neuroprobe_eval` plus three exact license/notice files; public-history exclusions are recorded in `RELEASE_MIGRATION_PLAN.md`

## Change-entry template

### YYYY-MM-DD — Short change title

- Change ID/commit subject:
- Files changed:
- Classification: portability-only | packaging | deletion-no-behavior-change | test/docs | computation-affecting | mixed
- Reason:
- Behavioral effect:
- Smoke cases affected:
- Before:
- After:
- Metric/config deltas:
- Evidence/report ID:
- Reviewer/disposition:
- Follow-up or user review needed:

## Changes

No post-baseline migration changes have been made yet.
