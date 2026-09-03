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

### 2026-08-30 — Reduce the public runtime to the retained paper surface

- Change ID/commit subject: `refactor: remove unsupported migration surface`
- Files changed: unsupported models/configs, archived preprocessors, obsolete/private launchers, retained launcher path defaults, runtime contract tests, `tests/source_manifest.json`, `provenance/surface/**`, and `CHANGELOG_PARITY.md`
- Classification: deletion/no-behavior-change for retained cases
- Reason: remove BRANT, unused baselines, archived preprocessing, cluster-only wrappers, and SEEGnificant-only coordinate support while preserving every paper-linked model/config family and BaRISTA.
- Behavioral effect: removed families and historical wrappers are no longer executable; retained launchers use the shipped `default` paths group unless callers set `PATHS_CFG`.
- Smoke cases affected: retained Logistic, MLP, CNN, linear, HTNet, PopT, BrainBERT, DIVER, and BaRISTA configuration/launcher surfaces.
- Before: the package shipped broken/private path defaults and unsupported historical runtime surface.
- After: retained and removed inventories are explicit, all retained launchers compose from public configs, and historical consulted hashes remain frozen in provenance.
- Metric/config deltas: none for retained cases; deleted unsupported cases intentionally have no runtime parity claim.
- Evidence/report ID: exhaustive `provenance/surface/{retained,removed}.tsv`; full CPU suite against pinned TorchBrain (72 passed, 7 expected BaRISTA/xformers skips).
- Reviewer/disposition: two implementation review/fix passes plus independent paper-provenance reachability review; broken BRANT launcher, stale path configs, source-manifest deviations, and environment guidance fixed before commit.
- Follow-up or user review needed: translate remaining launcher matrices into canonical declarative run manifests before any further script deletion.

### 2026-08-30 — Replace example-era onboarding with standalone public guidance

- Change ID/commit subject: `docs: add standalone iMINDBench workflow`
- Files changed: `imindbench/README.md`, `CHANGELOG_PARITY.md`
- Classification: test/docs
- Reason: replace stale TorchBrain-example, private-path, removed-leaderboard, and nonexistent-script instructions with the actual standalone package workflow.
- Behavioral effect: documentation only.
- Smoke cases affected: environment creation, Brainsets preparation, CPU Logistic invocation, output isolation, parity tooling, and GPU model prerequisites.
- Before: onboarding assumed the private monorepo layout and referenced removed or absent assets.
- After: commands use the canonical `imindbench` environment/CLI, merged public Brainsets API, caller-owned paths, and explicit parity limitations.
- Metric/config deltas: none.
- Evidence/report ID: link/path checks, Hydra composition, CLI/parity help checks, and post-cleanup model/config inventory review.
- Reviewer/disposition: two documentation review passes plus an independent operational review; the dependency publication was pending at this phase and was subsequently completed.
- Follow-up or user review needed: none; the final reviewed TorchBrain commit is public and artifact-validated below.

### 2026-08-30 — Validate the fresh application environment

- Change ID/commit subject: `fix: make fresh environment installation runnable`
- Files changed: `environment.yml`, `pyproject.toml`, `imindbench/conf/__init__.py`, `tests/test_imindbench_pipelines/test_imindbench_run_eval.py`, `RELEASE_MIGRATION_PLAN.md`, `CHANGELOG_PARITY.md`
- Classification: packaging/test/docs
- Reason: fresh installation exposed mixed-channel solver delay, an incorrect direct-dependency project name, and a non-importable Hydra config module.
- Behavioral effect: Conda resolution is restricted to conda-forge, the pinned warmup repository uses its published `warmup-scheduler` name, and installed CLI help can import packaged configs from an unrelated working directory.
- Smoke cases affected: fresh environment creation, editable installation, `pip check`, installed help, and the full CPU test suite.
- Before: full environment solving stalled; pip rejected the warmup dependency metadata; installed help failed with `MissingConfigException`.
- After: a fresh Python 3.10 environment installs TorchBrain and iMINDBench, `pip check` passes, imports resolve to the intended checkouts, and installed help succeeds from `/tmp`.
- Metric/config deltas: none.
- Evidence/report ID: PyTorch `2.9.1+cu128`; `torch.cuda.is_available() == False`; full fresh-environment suite with writable XDG cache (80 passed).
- Reviewer/disposition: boundary failures reproduced and fixed directly; packaging/config import and unrelated-CWD behavior rechecked.
- Follow-up or user review needed: completed by the later minimal-Conda/full-pip one-command validation below.

### 2026-08-31 — Run public Brainsets preparation smoke gates

- Change ID/commit subject: `test: record prepared-data smoke identities`
- Files changed: `config/brainsets_smoke_manifest.json`, `scripts/validate_brainsets_smoke.py`, `tests/test_neuroprobe_shared_artifacts.py`, `imindbench/README.md`, `RELEASE_MIGRATION_PLAN.md`, `CHANGELOG_PARITY.md`
- Classification: test/docs
- Reason: freeze the manifest IDs selected by real public preparation and update the dependency pin after fixing PIPPI's isolated import boundary.
- Behavioral effect: the smoke validator defaults to the checked-in recording for each dataset while retaining `--recording-id` overrides; evaluation algorithms are unchanged.
- Smoke cases affected: Neuroprobe2025/V2 `sub_1_trial001`, BYD `sub-CS48_ses-P48CSR1`, and PIPPI `sub-01_ses-iemu_task-film_acq-clinical_run-1`.
- Before: selected IDs were interactive and the PIPPI isolated pipeline failed before manifest discovery because a pure helper import triggered the Torch-dependent dataset package.
- After: all three `--list` gates pass; Neuroprobe and BYD single-recording preparation/load/idempotent reruns pass; all 16 PIPPI recordings prepare, load, and survive a full idempotent rerun.
- Metric/config deltas: none.
- Evidence/report ID: processed Neuroprobe SHA256 `a58777ed3b7191c239c764657177dd58eec794a87d9b00f2d8ca032a7519c19b`; BYD `9bd6c48eb7375dd799c8173c0f8499c54cafb640a62ee0f85eebb105fa37bae9`; selected PIPPI `f76e220facc682a0dcb45806cf6656d3d40b24075718e9f37b764d291ad2c6a1`; fresh suite 80 passed.
- Reviewer/disposition: PIPPI fix had two implementation passes and an independent wheel/API/isolation review; real preparation retry passed.
- Follow-up or user review needed: full Neuroprobe2025 preparation and cross-recording NeuroprobeV2 regime gates remain; TorchBrain commit `492f94a594e81d30ef38db32d8be145627421b0d` must be pushed before its public VCS pin resolves.

### 2026-08-31 — Complete full Neuroprobe and NeuroprobeV2 CPU gates

- Change ID/commit subject: `test: validate full Neuroprobe shared artifacts`
- Files changed: `scripts/validate_brainsets_smoke.py`, `tests/test_neuroprobe_shared_artifacts.py`, `RELEASE_MIGRATION_PLAN.md`, `CHANGELOG_PARITY.md`
- Classification: test/docs
- Reason: verify that one Neuroprobe2025 preparation corpus supports every public NeuroprobeV2 routing regime without a second pipeline or artifact mutation.
- Behavioral effect: adds an opt-in read-only `--neuroprobe-v2-regimes` validation mode; evaluation and preparation algorithms are unchanged.
- Smoke cases affected: folds 0 and 1, binary onset train/val/test for within-session, hold-in-session, hold-out-session, and hold-out-subject.
- Before: only one real explicit-recording Neuroprobe load and synthetic regime contracts were verified.
- After: all 26 Neuroprobe2025 recordings prepare and survive a full idempotent rerun; all 24 NeuroprobeV2 fold/regime/split selections materialize the interval and channel selectors for every resolved shared H5 recording. The validator closes each lazy H5 handle, confirms its read-only inventory is unchanged, and compares one fixed one-second neural window byte-for-byte through both public Neuroprobe views.
- Metric/config deltas: none.
- Evidence/report ID: 26/26 prepared; 24 selections and 166 recording-selection opens; per-fold V2 train recording counts 1/26/25/23 for within/hold-in/hold-out-session/hold-out-subject; selected H5 SHA256 remains `a58777ed3b7191c239c764657177dd58eec794a87d9b00f2d8ca032a7519c19b`; fixed-window SHA256 `6ced00f906e65914cace8cdbafab5928692528699a3ab672df6cd1d690d355b8` with shape `[2048, 155]`; final full suite 82 passed.
- Reviewer/disposition: exhaustive real-corpus validator pass plus strengthened multi-recording/fold/closure tests and two implementation review passes.
- Follow-up or user review needed: investigate the checkpoint-free BYD MLP metric discrepancy recorded below without changing the migrated runtime speculatively; broader per-provider deterministic-window/evaluation-output checks and optional full BYD preparation remain.

### 2026-08-31 — Run checkpoint-free CPU parity cases

- Change ID/commit subject: `test: record CPU parity evidence`
- Classification: test/docs
- Reason: smoke-test the migrated evaluation path against frozen paper-output records using freshly prepared public artifacts.
- Behavioral effect: none; no runtime code was changed in response to results.
- Smoke cases affected: NeuroprobeV2 Logistic/onset and BYD MLP/global-flow, folds 0 and 1.
- Results: Logistic passed the strict metric/config-record comparator. BYD MLP completed successfully but failed metric parity with matching identity/config hashes: test ROC-AUC was `0.3944888889` versus `0.4259555556` for fold 0 and `0.4892899308` versus `0.5817169843` for fold 1.
- Evidence/report ID: local fresh-output comparator reports under the validation run root; generated run data and reports remain untracked.
- Reviewer/disposition: flag as an unresolved provenance/runtime-data drift candidate. Do not tune or alter the copied implementation until the historical environment and prepared-H5 provenance can be established.
- Follow-up or user review needed: compare historical dependency/data fingerprints where available; otherwise retain the failure transparently as a known provenance gap.

### 2026-09-01 — Freeze deterministic HTNet GPU parity case

- Change ID/commit subject: `test: add deterministic HTNet parity case`
- Files changed: `artifacts/parity_reference/manifest.json`, `artifacts/parity_reference/cases/neuroprobev2_htnet500_hpf_global_onset_sub1_sess1.json`, `artifacts/parity_reference/README.md`, `tests/test_parity_tools.py`.
- Classification: test/docs
- Reason: provide the GPU machine with one checkpoint-free neural-network case used by the Figure 4 preprocessing comparison.
- Behavioral effect: adds command construction and metric/config-record comparison only; no evaluation implementation changes.
- Smoke case affected: NeuroprobeV2 within-session binary onset, subject 1/session 1, folds 0 and 1, deterministic HTNet 500 Hz with the 15-second-context HPF/global-robust preprocessor.
- Evidence/report ID: source result SHA256 `7166736abeeb89bd23b3eef552d53abcc3d93f4c891fdc1f4e7bbeb167e1b2b3`; reduced reference SHA256 `9fd1f0ae8d2f1940b0a7b2ff6ba11d22fb72978623a6fb18321e9dc88acc319a`; historical launcher SHA256 `2cbe3a2b1c19434539797e21136ac4faa6d9e1503b17ae29565ba3aac2c188e0`.
- Reviewer/disposition: source identity, canonical config/preprocessor hashes, fold metrics, deterministic overrides, generated command, and Hydra composition verified; parity-tool focused tests pass.
- Follow-up or user review needed: run the generated command on the GPU machine with full execution fingerprints and compare into a fresh report directory.

### 2026-09-02 — Establish exact NeuroprobeV2 MLP GPU parity

- Change ID/commit subject: `test: record NeuroprobeV2 MLP GPU parity`
- Files changed: `artifacts/parity_reference/manifest.json`, `artifacts/parity_reference/cases/neuroprobev2_mlp_multistft_onset_sub1_sess1.json`, `artifacts/parity_reference/README.md`, `tests/test_parity_tools.py`, `provenance/AUDIT.md`, `provenance/consulted_artifacts.tsv`, `provenance/surface/retained.tsv`, `imindbench/README.md`, `RELEASE_MIGRATION_PLAN.md`, `CHANGELOG_PARITY.md`.
- Classification: test/docs
- Reason: formalize the checkpoint-free MLP run that exactly reproduced the paper record on newly prepared public Neuroprobe data.
- Behavioral effect: adds one manifest-built command and immutable comparison record; no evaluation implementation changed.
- Smoke case affected: NeuroprobeV2 within-session binary onset, subject 1/session 1, folds 0 and 1, MLP with `laplacian_multi_stft_2048Hz`.
- Results: every train/validation/test accuracy and ROC-AUC metric matched exactly at `1e-9` after restoring the historical profile of four persistent pinned workers, prefetch factor two, and six preprocessing threads.
- Evidence/report ID: historical result SHA256 `2abf4d1873bf4b703ce02fbf3e1ba565afbdb23b96ba87cb657f02e7e609fbb7`; candidate SHA256 `003182f61a09aee2bbc0ab7820c6c1cf02e75e6902085a81184c1c9cd3d31bd4`; resolved-config SHA256 `82055c8c6dd2c453f9054c6a67bfa5f0ab1ffe0175efd607069cde741283827d`; run-log SHA256 `65ef53eddc64e014d29a2ac82b2674689d25f4de8e678f57c91682b85e4ede43`; fresh comparator status `PASS`.
- Reviewer/disposition: promote this MLP case to completed development GPU parity evidence; retain deterministic HTNet as an optional runnable reference. Built-artifact release acceptance was pending at this phase and is completed in the entry below.
- Follow-up or user review needed: completed by the later non-editable built-artifact reproduction.

### 2026-09-02 — Remove the final private TorchBrain preparation fallback

- Change ID/commit subject: `fix: remove private BYD label fallback` (`torch_brain` `e39f48ce0ec8c8f59be2507dca8ae172cce79d28`)
- Files changed: TorchBrain BYD pipeline and portability tests; iMINDBench dependency pin, smoke manifest, onboarding, and release plan.
- Classification: runtime/test/docs
- Reason: the direct BYD processor retained a developer-specific label fallback even though the normal public pipeline already used packaged labels.
- Behavioral effect: `process_file(..., labels_dir=None)` now resolves packaged labels through the canonical validator; explicit paths retain their prior behavior. Opt-in real-data tests use `BYD_CS41_RAW_DIR` rather than a private fixture path.
- Evidence/report ID: 226 migrated dataset/pipeline tests passed with two expected real-data skips; full TorchBrain Ruff check and format check passed; tracked private-path scan is empty.
- Reviewer/disposition: two review/fix passes found no behavioral or packaging regression; the public feature branch contains the immutable dependency commit.
- Follow-up or user review needed: verify packaged resources and imports from fresh sdist-derived wheels.

### 2026-09-02 — Validate installed TorchBrain wheel provenance

- Change ID/commit subject: `fix: validate installed TorchBrain artifacts`
- Files changed: `config/brainsets_smoke_manifest.json`, `scripts/validate_brainsets_smoke.py`, `tests/test_neuroprobe_shared_artifacts.py`, `RELEASE_MIGRATION_PLAN.md`, `CHANGELOG_PARITY.md`.
- Classification: runtime/test/docs
- Reason: fresh artifact testing exposed that the smoke validator recognized clean Git checkouts and VCS installs but not a reviewed wheel installation.
- Behavioral effect: smoke-manifest schema 2 pins the TorchBrain wheel SHA256. Validation accepts either the exact clean source checkout or the pinned non-editable wheel, binds distribution metadata to the imported module, rejects dirty SCM versions/shadow imports, and verifies installed package files against installed `RECORD` hashes.
- Evidence/report ID: TorchBrain wheel SHA256 `b2c318141f23bce4ff2749fd5e9000fd679760d735c02af9e8cc5557a140172d`; 21 focused source/artifact tests plus the full iMINDBench suite pass.
- Reviewer/disposition: two review/fix passes closed unchecked-hash, nested-environment, untracked-file, shadow-import, and installed-file-integrity gaps.
- Follow-up or user review needed: none; the unrelated-directory installed-wheel suite and loader workflow passed.

### 2026-09-02 — Reproduce MLP parity from release artifacts

- Change ID/commit subject: `test: validate built release artifacts`
- Files changed: `tests/test_neuroprobe_shared_artifacts.py`, `artifacts/parity_reference/README.md`, `RELEASE_MIGRATION_PLAN.md`, `CHANGELOG_PARITY.md`.
- Classification: test/docs
- Reason: close the release migration plan's non-editable built-artifact acceptance gate using clean clones of the requested iMINDBench and TorchBrain-public repositories.
- Behavioral effect: no evaluation or preparation behavior changed; additional tests cover legacy archive metadata and installed-file drift detection.
- Smoke cases affected: all four migrated public loaders and the NeuroprobeV2 MLP/multi-STFT onset GPU case.
- Results: a disposable environment installed fresh sdist-derived wheels with no editable packages, passed `pip check`, import/CLI/resource validation, unchanged-inventory loader smokes, and the full iMINDBench suite. The CUDA MLP run again matched every historical metric for both folds at `1e-9`.
- Evidence/report ID: source commits iMINDBench `302b66073a201823639700b284579db9ba7390ff` and TorchBrain `e39f48ce0ec8c8f59be2507dca8ae172cce79d28`; iMINDBench sdist/wheel SHA256 `72226acaf88371d31092b44e825008d9e00e1a75a620df4eefc49bec735978dd` / `9fae43948f412cb19dc9124b86d9b6612f4a9c825b1789cb86d5e271cb99d5bb`; TorchBrain sdist/wheel SHA256 `b24920bc6a2bedb8b206270bd7fd7da70e6e3eb92440d63c894e2b9aa10d7fbf` / `b2c318141f23bce4ff2749fd5e9000fd679760d735c02af9e8cc5557a140172d`; candidate SHA256 `40e0f03ecc67e9ef411cb8a46969b750781fa1325bd056ba61bfdf2cdf4f69f6`; comparator report SHA256 `51b38dad9eb67568ebbdb4c84b5c2e03b443a0ef9eed2ec2612fc165d00bd935`, status `PASS`.
- Reviewer/disposition: two independent simplification/bug-risk review cycles completed; no blocking artifact, source-identity, packaging, or parity finding remains.
- Follow-up or user review needed: the one-command environment gate is completed below. Push is intentionally deferred for this local-only continuation.

### 2026-09-02 — Make clean environment creation practical

- Change ID/commit subject: `fix: simplify release environment solve`
- Files changed: `environment.yml`, `RELEASE_MIGRATION_PLAN.md`, `CHANGELOG_PARITY.md`.
- Classification: packaging/docs
- Reason: both the default and explicit libmamba paths spent more than ten CPU-active minutes solving the broad Conda scientific matrix, although the validated environment already obtained those packages from PyPI.
- Behavioral effect: Conda now owns only Python 3.10, pip, and setuptools; the existing bounded NumPy/SciPy/pandas/sklearn/h5py/MNE/Hydra constraints move unchanged into the pip subsection beside the exact PyTorch/xformers pins.
- Results: `conda env create --prefix <fresh-prefix> --file environment.yml` completed; Conda resolution finished in under 30 seconds and the full cached installation completed in roughly two minutes. Installing the reviewed iMINDBench and TorchBrain wheels produced no editable packages, passed `pip check`, resolved both imports from site-packages, reported PyTorch `2.9.1+cu128`/CUDA `12.8`/xformers `0.0.33.post2`, passed CLI help, and passed all 91 tests.
- Reviewer/disposition: two independent reviews agreed this is the smallest maintainable change and matches the established parity environment; exact Conda build pins were rejected as brittle and unnecessary.
- Follow-up or user review needed: none for the local release-migration gate. A generated pip constraints/lock artifact may be added later if byte-level cross-date dependency resolution is required.

### 2026-09-03 — Test the Neuroprobe historical profile on BYD MLP

- Change ID/commit subject: `test: record BYD historical-profile diagnostic`
- Files changed: `RELEASE_MIGRATION_PLAN.md`, `CHANGELOG_PARITY.md`.
- Classification: experiment evidence/docs
- Reason: determine whether the four-worker, pinned/persistent-loader, prefetch-two, six-preprocessing-thread profile that restored exact NeuroprobeV2 MLP parity also explains the BYD MLP drift.
- Behavioral effect: none; the profile was applied only as a caller-owned GPU diagnostic and the BYD manifest defaults were not changed.
- Results: strict comparison remained `FAIL`. Fold 0 test accuracy/ROC-AUC was `0.4133333333`/`0.3784888889` versus historical `0.44`/`0.4259555556`; fold 1 was `0.5779220779`/`0.5952099848` versus historical `0.5519480519`/`0.5817169843`. The profile improved fold 1 beyond the historical score but did not restore record identity.
- Evidence/report ID: caller-owned run `byd_mlp_historical_profile_20260903`; result SHA256 `0ad3c293101bb92688635a3744f8825e3e335f25ac583952ad25f1922ebee6c0`; resolved-config SHA256 `f2d5c88efa7b84d9c10829003783f24ff9cde3c3eceae840ae987328db7a3d7d`; run-log SHA256 `95ba0fdeaa1e6debe1bff617e4981e7e8a20c431c3994bc02977fe0b9de00d73`; comparator-report SHA256 `f435434d288ceceb57cc0dcdce9916b76fbd6050a746ac89999b914ee3cb0e7f`, status `FAIL`.
- Reviewer/disposition: retain BYD as a transparent data/environment provenance drift finding. The Neuroprobe profile is dataset/run-specific evidence, not a global MLP setting.
- Follow-up or user review needed: recover historical BYD H5/environment fingerprints before attributing the remaining difference or changing evaluation code.

### 2026-09-03 — Add BYD and PIPPI Logistic parity diagnostics

- Change ID/commit subject: `test: add provider Logistic parity evidence`
- Files changed: two reduced Logistic reference records, parity manifest/tests, parity READMEs, provenance inventories, release plan, and this changelog.
- Classification: experiment evidence/test/docs
- Reason: use deterministic CPU Logistic runs to distinguish BYD/PIPPI data-path drift from the neural-training discrepancy.
- Behavioral effect: none; two paper-referenced runnable cases and their command/comparison tests were added without changing evaluation code.
- Smoke cases affected: BYD full-tier global-flow subject 41/session 1 and PIPPI high-cov speech subject 1/session 1, both multi-STFT within-session with folds 0 and 1.
- Results: both identities/config hashes and all historical split counts/feature shapes matched, but strict metrics failed. BYD candidate test AUCs were `0.3699555556`/`0.5471411705` versus `0.3745777778`/`0.5322988700`; PIPPI was `0.6150793651`/`0.7654478458` versus `0.6169217687`/`0.7704081633`.
- Evidence/report ID: source result SHA256 values `6467500145bd4bbdc950b83f1d2b51b06c8bdfdf3076d76100ae54697f769772` and `10c24e3fa0be7e972539f165c921983e51cdf0da565dd5cb968a803cb93f32c1`; candidate SHA256 values `da23d2984451475f23b9f07e72705bfc52a0bca368c4435491d93dcaa3307247` and `fc05207467f495851c3603606143fd36a6678252b7632c25c24e8fa6270af897`; combined report SHA256 `229b4f9aeaf4e0f50bd60703dec32e54c76830c27f81240d2ede7e9e370ce0b6`, status `FAIL`.
- Reviewer/disposition: retain both as transparent strict-parity failures. BYD's previously audited raw payload and all 43 PIPPI neural/channel/selected-split arrays compared here are byte-identical across historical and migrated H5s, narrowing likely causes to post-load numeric/runtime or unrecoverable historical working-tree differences.
- Follow-up or user review needed: only pursue post-preprocessing tensor hashes or historical scikit-learn/BLAS reconstruction if strict provider-specific metric identity is required; do not tune the public defaults to these two records.

### 2026-09-03 — Identify the BYD/PIPPI window-indexing parity cause

- Change ID/commit subject: `docs: record historical slicing semantics`
- Files changed: `provenance/AUDIT.md`, `RELEASE_MIGRATION_PLAN.md`, `CHANGELOG_PARITY.md`.
- Classification: experiment evidence/docs
- Reason: explain why byte-identical provider payloads and matching historical runtime settings still produced different Logistic metrics after migration.
- Behavioral effect: none; the investigation used isolated diagnostics and did not modify TorchBrain, processed H5 files, or existing outputs.
- Results: the paper environment's `temporaldata==0.1.1` lazy slicer used floor-based boundary conversion, while migrated TorchBrain uses grid snapping plus ceil. The resulting one-sample window shift changes every prepared feature tensor. Running the original evaluation package under `tb_buildathon`, and independently restoring floor indexing only for an iMINDBench diagnostic, reproduced all historical BYD and PIPPI Logistic metrics exactly in both folds.
- Evidence/report ID: caller-owned, non-overwriting runs `original_scripts_tb_buildathon_20260903` and `imindbench_legacy_floor_slice_20260903` under `data/imindbench_validation/runs/`.
- Reviewer/disposition: record this separately from the historical worker/thread profile that restored NeuroprobeV2 MLP parity. The slicing change originated in TemporalData commit `ad3e9850b5103bb8cc4eca421431b65658563fcf` on 2026-03-16, reached v0.1.2 on 2026-05-29, and was incorporated into TorchBrain by merge `6069a76c84bcbc96eb9253d1263e032046853e8f`.
- Follow-up or user review needed: decide whether parity requires an explicit legacy window-indexing policy at the iMINDBench evaluation boundary; do not change TorchBrain globally.

### 2026-09-03 — Reproduce coordinate-aware PopT-v2 provider cases

- Change ID/commit subject: `test: record PopT-v2 provider parity`
- Files changed: parity manifest/tests/README, package README, provenance audit, release plan, consulted-artifact inventory, and this changelog.
- Classification: experiment evidence/test/docs
- Reason: test the temporal slicing finding with the paper's pretrained, coordinate-aware PopT-v2 model rather than only Logistic regression.
- Behavioral effect: no runtime code changed; PIPPI PopT moved from `NOT-COMPARABLE` to checkpoint-bound `RUNNABLE` because its historical resolved config names the retained checkpoint and both copies have SHA256 `cf4e835d5309559d468b2f1ebd9b76882398c30bedae6c0c8bc6fdb4c506b52f`.
- Results: BYD global-flow subject 41/session 1 and PIPPI speech subject 1/session 1 reproduced every historical train/validation/test accuracy and ROC-AUC metric exactly in both folds with legacy floor slicing. Logs confirm numerical LIP coordinates reached `MultiSubjBrainPositionalEncoding`; BYD dropped six non-finite coordinate channels and retained the historical 74-channel shape.
- Evidence/report ID: caller-owned, non-overwriting run `imindbench_popt_main_ckpt_legacy_floor_coords_20260903`; BYD result/config/log SHA256 `cc2b0ea4f415416f299432fffa1a74f24156599575069197d6be428f72c9cd91` / `f9eeafa96653ccd6ca5baf6e6cb9e8702c862f5fe5fc031cf43d7e0c2d5e6aac` / `00a38c9584f9ac911b5abfad17c84c36e3afedb2b3a0566d66775bad9fd6bd5d`; PIPPI result/config/log SHA256 `f4a01e9d125711b73fb52d3381f28b80ab9e921075ce3415cf7968d4efd8068f` / `ef9f1c8ee7d371e9c67313f702d40f050bb1ce947a12a371c24c218d1880f560` / `c3f9573726c81118bf05d4945030a74b3be50468418d0be19fb4460f0e38110a`; PIPPI comparator report JSON/Markdown SHA256 `42261b375edc7fd941f959aad49afa4be89e806bec09caf4a037c8a81724ec96` / `65328fdd253a70723d0487fd3bf176afaf160a71eb7c0c89017164b30f9b285f`, status `PASS`.
- Reviewer/disposition: accept the same pretrained checkpoint used by the main PopT-v2 results for this parity case; keep the legacy slicing policy scoped to iMINDBench rather than reverting TorchBrain globally.
- Follow-up or user review needed: implement and test an explicit legacy window-indexing policy before making the diagnostic path a supported release command.
