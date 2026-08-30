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

### 2026-08-30 — Port retained source tests

- Change ID/commit subject: `test: port retained neuroprobe evaluation coverage`
- Files changed: `tests/source_manifest.json`, `tests/test_neuroprobe_eval_pipelines/**`
- Classification: test/docs
- Reason: preserve high-value BaRISTA, legacy-model, runner, and preprocessor coverage before simplification.
- Behavioral effect: no runtime change; six tests are byte-identical to source and the BaRISTA test changes only its config-root path for the repository layout.
- Smoke cases affected: retained baseline unit/integration gates.
- Before: relevant tests existed only in the TorchBrain source repository.
- After: 25 tests pass; 7 BaRISTA implementation tests skip when `xformers` is unavailable.
- Metric/config deltas: none.
- Evidence/report ID: `tests/source_manifest.json`; `conda run -n tb_buildathon python -m pytest -q -rs tests/test_neuroprobe_eval_pipelines`
- Reviewer/disposition: root reviewed; accepted.
- Follow-up or user review needed: rerun skipped BaRISTA tests in the final `imindbench` environment with `xformers` installed.
