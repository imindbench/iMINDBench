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

### 2026-08-30 — Add baseline packaging and environment contract

- Change ID/commit subject: `build: package baseline evaluation runtime`
- Files changed: `pyproject.toml`, `environment.yml`, `.gitignore`
- Classification: packaging
- Reason: make the imported baseline installable and define the canonical Python 3.10 `imindbench` environment before the namespace migration.
- Behavioral effect: adds distribution metadata, package-data inclusion, console entrypoint, dependency ownership, and generated-artifact exclusions; runtime algorithms are unchanged.
- Smoke cases affected: package build/import and all later installed-package runs.
- Before: the imported example was not a standalone root package.
- After: a no-dependency local wheel build includes 95 Hydra YAMLs, six stable JSONs, and all three required license/notice files.
- Metric/config deltas: none.
- Evidence/report ID: successful local wheel inspection (the recorded wheel hash is a one-build artifact ID because ZIP timestamps make ordinary wheel bytes non-reproducible).
- Reviewer/disposition: root reviewed; accepted for migration, with release dependency pins still subject to final fresh-environment validation.
- Follow-up or user review needed: replace the development editable `torch_brain-public` install with an immutable public release/commit and rerun the built-wheel acceptance gate.

### 2026-08-30 — Freeze paper-figure provenance inventory

- Change ID/commit subject: `docs: inventory paper-figure run provenance`
- Files changed: `provenance/**`
- Classification: test/docs
- Reason: map the scoped paper-figure notebooks to immutable source artifacts and candidate parity cases before runtime simplification.
- Behavioral effect: none; analysis-only provenance, with no notebook execution or runtime changes.
- Smoke cases affected: five proposed cases spanning NeuroprobeV2, BYD, PIPPI, Logistic, MLP, PopT, BaRISTA, within-session, and hold-in routing.
- Before: notebook/YAML/output relationships existed only in the dirty TorchBrain checkout.
- After: 14 notebooks, 23 active YAMLs, 102 output roots, and 77 consulted artifact hashes/statuses are recorded under source-relative identifiers.
- Metric/config deltas: none.
- Evidence/report ID: `provenance/paper_figs_provenance.json`, `provenance/consulted_artifacts.tsv`, `provenance/AUDIT.md`.
- Reviewer/disposition: two agent review passes and root schema/path review; accepted with documented provenance gaps.
- Follow-up or user review needed: recover/freeze the exact BaRISTA checkpoint/launcher provenance before claiming strict BaRISTA reproduction.

### 2026-08-30 — Correct baseline release metadata

- Change ID/commit subject: `fix: correct baseline package contracts`
- Files changed: `pyproject.toml`, `environment.yml`, `.gitignore`, `CHANGELOG_PARITY.md`
- Classification: packaging
- Reason: address verified package-review findings before namespace migration.
- Behavioral effect: removes the inaccurate whole-distribution Apache-only license claim while retaining all component notices; packages shell launchers; adds the test dependency contract; and restricts generated artifacts to reviewed parity references.
- Smoke cases affected: wheel metadata/content, fresh-environment tests, and artifact hygiene.
- Before: mixed-license code was labeled solely Apache, launchers/tests were missing from the installed contract, and arbitrary artifact files were trackable.
- After: component license files remain authoritative, launchers are wheel data, pytest is declared, and only `artifacts/parity_reference/**` is allowlisted.
- Metric/config deltas: none.
- Evidence/report ID: independent package review after commits `d108a28` and `3f1b368`.
- Reviewer/disposition: root verified and fixed all packaging-only findings.
- Follow-up or user review needed: migrate `brainsets.datasets` to `torch_brain.datasets` in its dedicated runtime/import commit; clean current README launcher/environment claims during namespace/script cleanup.

### 2026-08-30 — Rename the installed package namespace

- Change ID/commit subject: `refactor: rename neuroprobe_eval package to imindbench`
- Files changed: `neuroprobe_eval/**` → `imindbench/**`, `tests/test_neuroprobe_eval_pipelines/**` → `tests/test_imindbench_pipelines/**`, `tests/source_manifest.json`, `pyproject.toml`, `THIRD_PARTY.md`, `CHANGELOG_PARITY.md`
- Classification: portability-only
- Reason: give the standalone distribution one canonical installed namespace independent of its TorchBrain example origin.
- Behavioral effect: imports, module launch commands, packaging discovery/data paths, current documentation, and test destinations use `imindbench`; evaluation algorithms and configuration values are unchanged.
- Smoke cases affected: package import, Hydra config discovery, launcher entrypoints, and retained tests.
- Before: the `imindbench` distribution installed and launched the `neuroprobe_eval` Python package.
- After: the distribution installs and launches `imindbench`; the old package namespace is absent.
- Metric/config deltas: none; W&B project identity, cache directory names, provider/model/task identifiers, stable JSONs, and source-provenance URIs remain unchanged.
- Evidence/report ID: static namespace scan, retained pytest suite, built-wheel content inspection, and unrelated-working-directory import provenance check.
- Reviewer/disposition: two independent reviews completed; canonical console name and test-manifest deviation findings fixed before commit.
- Follow-up or user review needed: make bare Hydra `--help` compose without required experiment overrides in a separate usability change; Brainsets imports, README cleanup, and script reduction remain separate changes.

### 2026-08-30 — Use the merged TorchBrain dataset API

- Change ID/commit subject: `fix: load datasets from torch_brain public API`
- Files changed: `imindbench/utils/pipeline_contracts.py`, `tests/test_imindbench_pipelines/test_torch_brain_dataset_imports.py`, `CHANGELOG_PARITY.md`
- Classification: portability-only
- Reason: standalone Brainsets is not installed; the merged public loaders are exported by `torch_brain.datasets`.
- Behavioral effect: processed providers lazily resolve the same four dataset classes from the supported public namespace and fail with updated actionable errors.
- Smoke cases affected: all Neuroprobe2025, NeuroprobeV2, BYD, and PIPPI processed-provider runs.
- Before: a fresh environment failed on the removed `brainsets.datasets` namespace.
- After: all provider loaders resolve from `torch_brain.datasets`; no standalone Brainsets import remains.
- Metric/config deltas: none.
- Evidence/report ID: focused provider import test plus retained pipeline suite.
- Reviewer/disposition: two independent reviews completed; root verified against torch_brain-public commit `c9fe75a3a0fa1eaec29314248cf3e0ae0e18e05c` API.
- Follow-up or user review needed: pin the public TorchBrain distribution/commit in release metadata and run real loader smoke tests after preparation.

### 2026-08-30 — Make bare CLI help composable

- Change ID/commit subject: `fix: make installed CLI help self-contained`
- Files changed: `imindbench/run_eval.py`, `pyproject.toml`, `tests/test_imindbench_pipelines/test_imindbench_run_eval.py`, `CHANGELOG_PARITY.md`
- Classification: portability-only
- Reason: Hydra previously failed before displaying help because required experiment groups had no defaults.
- Behavioral effect: `-h`/`--help` injects composition-only example groups when absent; non-help runs retain the required explicit experiment selections.
- Smoke cases affected: installed `imindbench --help` and `python -m imindbench.run_eval --help`.
- Before: bare help exited with `ConfigCompositionException`.
- After: bare help composes and displays available configuration groups without accessing data or launching a run.
- Metric/config deltas: none for experiment execution.
- Evidence/report ID: focused CLI wrapper tests and unrelated-working-directory wheel help check.
- Reviewer/disposition: two independent reviews completed; public `sys.argv` restoration and non-help preservation were added before commit.
- Follow-up or user review needed: none.

### 2026-08-30 — Freeze paper parity references and offline tooling

- Change ID/commit subject: `feat: add manifest-driven parity checks`
- Files changed: `artifacts/parity_reference/**`, `scripts/parity_tools.py`, `tests/test_parity_tools.py`, `CHANGELOG_PARITY.md`
- Classification: test/docs
- Reason: turn the reviewed paper-figure provenance into redistributable reference contracts and explicit, caller-mapped parity workflows.
- Behavioral effect: adds dry-run command construction and offline JSON comparison only; no experiment is launched and evaluation runtime is unchanged.
- Smoke cases affected: NeuroprobeV2 Logistic, BYD MLP, PIPPI PopT, NeuroprobeV2 BaRISTA, and NeuroprobeV2 hold-in PopT.
- Before: five candidates and source result hashes were documented, but there was no frozen metric record or executable comparison boundary.
- After: all five reduced reference records are byte-hashed; Logistic/MLP can report metric/config-record PASS/FAIL, while otherwise matching PopT/BaRISTA records report NOT-COMPARABLE until exact historical checkpoint hashes are recovered. Checkable mismatches always FAIL.
- Metric/config deltas: none; the records preserve historical identities, folds, and metrics verbatim.
- Evidence/report ID: `artifacts/parity_reference/manifest.json`; `conda run -n tb_buildathon python -m pytest -q tests/test_parity_tools.py` (17 passed), including adversarial hash/path/numeric/status/atomicity cases.
- Reviewer/disposition: independent review findings fixed; two follow-up local review passes completed for boundary validation, status semantics, atomic report finalization, provenance-claim limits, simplification, and private-path exclusion.
- Follow-up or user review needed: recover immutable PopT and BaRISTA reference checkpoint hashes before enabling strict comparison for those three cases; experiments remain unexecuted.

### 2026-08-30 — Add CPU Brainsets contract and artifact smoke checks

- Change ID/commit subject: `test: add Brainsets public contract smoke checks`
- Files changed: `config/brainsets_smoke_manifest.json`, `scripts/validate_brainsets_smoke.py`, `tests/test_brainsets_public_contract.py`, `tests/test_neuroprobe_shared_artifacts.py`, `CHANGELOG_PARITY.md`
- Classification: test/docs
- Reason: verify the public preparation interface and prepared-artifact loading boundary before running network or GPU experiments.
- Behavioral effect: adds read-only validation tooling and tests only; it does not prepare, download, or modify datasets.
- Smoke cases affected: Brainsets CLI IDs and isolated `uv` construction, packaged BYD/PIPPI labels, and Neuroprobe2025/NeuroprobeV2 shared-artifact loading.
- Before: public loader and preparation assumptions were documented but not executable from iMINDBench.
- After: the pinned TorchBrain source, manifest schema, packaged assets, selector contracts, deterministic recording access, and H5 immutability are checked explicitly; full-corpus hashing is opt-in.
- Metric/config deltas: none.
- Evidence/report ID: `conda run -n tb_buildathon env PYTHONPATH=/home/geeling/Projects/ieeg_project/torch_brain-public python -m pytest -q tests/test_brainsets_public_contract.py tests/test_neuroprobe_shared_artifacts.py` (21 passed).
- Reviewer/disposition: two implementation review/fix passes plus an independent review; parser optionality and real `Pipeline.process()` routing were added before commit.
- Follow-up or user review needed: push TorchBrain commit `c9fe75a3a0fa1eaec29314248cf3e0ae0e18e05c` so the immutable pin is publicly reachable, then run the network-backed preparation matrix.
