# iMINDBench release migration and Brainsets validation

## Goal

Release a minimal, clean copy of `torch_brain/examples/neuroprobe_eval` that can
(1) rerun representative published evaluations, (2) check their result parity,
and (3) trace paper notebooks to the YAML configurations and result trees they
consume. The released workflow must depend only on public `torch_brain`, including
its `brainsets` CLI and `torch_brain.datasets` loaders.

## Current implementation status

This plan is audited against iMINDBench commit
`056ca22c14e2726858b94c71d14f2290c1d8966d` and the public TorchBrain dependency
commit `492f94a594e81d30ef38db32d8be145627421b0d`. Repository bootstrap, the filtered
baseline import, source-test inventory, installable packaging, the `imindbench`
namespace migration, paper-figure provenance, frozen reduced reference records,
manifest-driven parity tooling, public Brainsets contract tests, retained-surface
reduction, and standalone onboarding are committed.

Subsequent CPU validation created a fresh `imindbench` environment, passed the
full suite, prepared and reloaded one recording for each provider, and completed
full PIPPI preparation plus an idempotent full rerun. No new evaluation or GPU
experiment has run. The next gates are full Neuroprobe2025 preparation,
cross-recording NeuroprobeV2 regime checks, and metric/config-record comparison
for the runnable Logistic and MLP cases; then the GPU handoff gate below. PopT and BaRISTA remain
`NOT-COMPARABLE` until the exact historical checkpoint hashes are established.

## 1. Validate the public Brainsets workflow

Use a fresh Conda environment rather than either existing `tb_buildathon`
environment. iMINDBench is a minimal release of `neuroprobe_eval`, whose current
canonical setup is Conda because it owns a compiled scientific/PyTorch stack and
GPU evaluation runtime. Retain its Python 3.10 contract and name the new canonical
environment `imindbench`:

```bash
cd /home/geeling/Projects/ieeg_project/iMINDBench
conda env create -f environment.yml
conda activate imindbench
python -m pip install -e /home/geeling/Projects/ieeg_project/torch_brain-public
```

Build `iMINDBench/environment.yml` by minimizing the existing
`torch_brain/examples/neuroprobe_eval/environment.yml`, preserving only packages
reached by the retained runtime, parity scripts and provenance tooling. During
migration, install `torch_brain-public` editable as shown above. For the public
release, the current reviewed source pin is
`492f94a594e81d30ef38db32d8be145627421b0d`; install that immutable Git commit (or
a release artifact proven to contain it) and record the artifact hash. Until the
user-authorized push gate is cleared, use the reviewed local checkout for
development and treat the public immutable installation as blocked.
Do not install standalone `brainsets`: its CLI, pipelines, data structures and
dataset loaders now come from `torch_brain-public`.

### Dependency ownership

iMINDBench directly uses `torch_brain-public` for the merged Brainsets CLI and
dataset classes. Those classes transitively use `torch_brain.data`, dataset base
classes/mixins, H5 serialization, intervals/time series, channel metadata, and
pipeline utilities, so install the complete public package rather than copying
selected TorchBrain modules.

The `imindbench` Conda environment owns the evaluation runtime: Python 3.10,
PyTorch 2.9.1 for the target release stack, NumPy/SciPy/pandas/h5py, scikit-learn,
Hydra/OmegaConf, MNE, retained model dependencies, and provenance tooling. Do not
retain Jupyter/plotting dependencies unless retained non-notebook code imports
them.
Install `torch_brain-public` without `[dev]`; its development extra requests
`torch~=2.0`, which must not replace the evaluation stack's parity-pinned PyTorch.
The package installation supplies its declared runtime dependencies, including
the `brainsets` CLI, uv, Ray, Click and PyNWB.

After every clean environment creation or dependency change, validate ownership
and resolution:

```bash
conda run -n imindbench python -m pip check
env -u PYTHONPATH conda run -n imindbench python -c "import torch, torch_brain; print(torch.__version__); print(torch_brain.__file__)"
```

Require PyTorch to match the environment pin and `torch_brain.__file__` to point
to `torch_brain-public` during development (or the pinned public installation at
release). The retained iMINDBench runtime imports no standalone `brainsets`; its
dataset imports now use `torch_brain.datasets`.
Record the original runs' Python/PyTorch/CUDA/scikit-learn versions separately:
historical-stack parity and current-release-stack compatibility are different
results, and version drift must not be misreported as a code-migration regression.

iMINDBench is installable from its root `pyproject.toml`; install it editable
during development:

```bash
conda run -n imindbench python -m pip install -e /home/geeling/Projects/ieeg_project/iMINDBench
```

Do not rely on running from `torch_brain/examples` or adding that directory to
`PYTHONPATH`. Run release commands from the iMINDBench checkout (or an unrelated
temporary directory) and add an import-provenance gate:

```bash
env -u PYTHONPATH conda run -n imindbench python -c "import imindbench; print(imindbench.__file__)"
env -u PYTHONPATH conda run -n imindbench python -m imindbench.run_eval --help
```

The equivalent `neuroprobe_eval` commands were reserved for the untouched
pre-rename baseline gate; that experiment was not run and must not now be
recreated from a TorchBrain checkout. Current validation uses `imindbench`. The
resolved module path must be inside the iMINDBench checkout, and validation must
fail if it resolves to a TorchBrain checkout, another worktree, or a stale
site-packages copy.

Conda owns only the iMINDBench application environment. `brainsets prepare`,
although launched from `imindbench`, intentionally invokes its bundled `uv` to
create a separate isolated Python 3.11 environment from each pipeline's inline
metadata. Do not pass `--use-active-env`: Neuroprobe2025, BYD and PIPPI pin
different preparation dependencies, and those must not leak into the evaluation
environment.

Use this same fresh `imindbench` environment for CPU smoke tests, parity runs,
notebook/YAML provenance auditing and the final CUDA/GPU acceptance case. Keep
large validation data outside the repository:

- raw downloads: `/home/geeling/Projects/ieeg_project/data/imindbench_validation/raw/`
- processed H5 files: `/home/geeling/Projects/ieeg_project/data/imindbench_validation/processed/`
- small logs/reports: `iMINDBench/artifacts/brainsets_smoke/`

Run these commands first with `--list`, then with one stable manifest ID via
`--single`; run the complete dataset only after the single-item gate passes:

```bash
conda run -n imindbench brainsets prepare neuroprobe_2025 --list --raw-dir /home/geeling/Projects/ieeg_project/data/imindbench_validation/raw --processed-dir /home/geeling/Projects/ieeg_project/data/imindbench_validation/processed
conda run -n imindbench brainsets prepare keles_byd_2024 --list --raw-dir /home/geeling/Projects/ieeg_project/data/imindbench_validation/raw --processed-dir /home/geeling/Projects/ieeg_project/data/imindbench_validation/processed
conda run -n imindbench brainsets prepare berezutskaya_pippi_2022 --list --raw-dir /home/geeling/Projects/ieeg_project/data/imindbench_validation/raw --processed-dir /home/geeling/Projects/ieeg_project/data/imindbench_validation/processed
```

For each dataset, copy one ID printed by `--list` into the corresponding command
and replace `--list` with `--single <manifest_id>`; keep every other option
unchanged. This makes the selected recording explicit and avoids transient IDs in
the checked-in plan. Once selected, check the stable manifest IDs and expected
recording IDs into the smoke manifest rather than selecting them interactively on
future runs.

The current `config/brainsets_smoke_manifest.json` schema pins TorchBrain commit
`492f94a`, the three preparation pipeline IDs, and four public dataset
class/directory/recording mappings (including NeuroprobeV2's shared directory).
`scripts/validate_brainsets_smoke.py` is a CPU-only, read-only post-prepare
checker: `--list-datasets` validates the source pin and lists mappings; `--root`
and `--dataset` inventory prepared H5 files and verify the pinned public loader;
optional `--recording-id` overrides the selected recording and `--full-hash`
hashes the complete corpus. It does not download or prepare data.

After the single-recording gates pass, full preparation is mandatory for PIPPI
and Neuroprobe2025:

```bash
conda run -n imindbench brainsets prepare berezutskaya_pippi_2022 --raw-dir /home/geeling/Projects/ieeg_project/data/imindbench_validation/raw --processed-dir /home/geeling/Projects/ieeg_project/data/imindbench_validation/processed
conda run -n imindbench brainsets prepare neuroprobe_2025 --raw-dir /home/geeling/Projects/ieeg_project/data/imindbench_validation/raw --processed-dir /home/geeling/Projects/ieeg_project/data/imindbench_validation/processed
```

Run each full command again without `--reprocess` to test safe reuse/idempotency.
BYD may remain a single-recording preparation gate initially, followed by a full
run when storage and runtime permit.

Make idempotency measurable: after the first run capture the sorted raw/processed
inventory, sizes, H5 SHA256s and mtimes plus stdout/stderr and the detected
TorchBrain spec used by uv. After the second run require no duplicate/unexpected
files, byte-identical H5s, no unexpected rewrites, and successful loader reopening.

At public TorchBrain commit `492f94a`, BYD and PIPPI resolve their packaged label
resources when `--labels-dir` is omitted. The option remains an explicit override,
not a normal preparation requirement. The remaining release gate is to verify the
packaged labels and brain-area tables from built wheel and sdist artifacts and
record their hashes in preparation provenance.

Expected intermediates are downloaded source assets under
`raw/<brainset_id>/`, followed by processed recording H5 files under
`processed/<brainset_id>/`. Each H5 must contain the neural signal, recording
metadata, task-label intervals and split selectors; BYD/PIPPI must also contain
channel coordinates and brain-area metadata. Capture manifest IDs, commands,
versions, file inventories and validation results in the small report directory.

After preparation, instantiate every public loader from `torch_brain.datasets`,
exercise every supported regime and `train`/`val`/`test` split, and fetch real
windows. Check shapes, timestamps, finite signal values, labels, split isolation,
channel IDs/coordinates/areas and deterministic repeated loading. Then run one
small Logistic evaluation per provider and require a valid `population_*.json`
under `iMINDBench/artifacts/brainsets_smoke/eval/<provider>/`.

NeuroprobeV2 intentionally has no separate preparation pipeline: it is a
different recording/regime-selection view over the H5 files produced by
`brainsets prepare neuroprobe_2025`. The V2 loader defaults to the same
`neuroprobe_2025/` processed directory and maps `within-session`,
`hold-in-session`, `hold-out-session`, and `hold-out-subject` onto the prepared
`within_session` split intervals and channel masks while changing which
recordings participate in training. Document this shared-artifact contract and
test all four V2 regimes immediately after the full Neuroprobe2025 preparation.
For folds 0 and 1, assert resolved recording IDs, the reused `within_session`
interval/channel-mask selectors, missing-selector failure, and identical fixed
recording/window bytes through the Neuroprobe2025 and NeuroprobeV2 loaders. Assert
that V2 creates no processed files.
Do not add `brainsets prepare neuroprobe_v2` unless V2 later requires materially
different processed files.

## 2. Create the minimal iMINDBench working copy

### Repository and source bootstrap

This bootstrap is complete. The repository uses the
`gc/neuroprobe-eval-public-migration` branch from initial `main` commit `6d9ccd0`,
and the audited import is recorded as the first migration commit. The historical
rule remains: the baseline commit contains only the explicit import allowlist,
never an indiscriminate `git add -A` snapshot.

The runtime/config/script baseline source is fixed to
`/home/geeling/Projects/ieeg_project/torch_brain-main` at tracked `main` commit
`726d96978c3e573c7a31e2e63b18c905e4c15d6b` (the BaRISTA release commit) from
`git@github.com:glchau/torch_brain.git`. Extract from that Git tree object, not its
filesystem working state. This deliberately includes the shared infrastructure
required by BaRISTA while excluding the later MVPFormer and SEEGnificant release
commits. At plan time it contains 335 tracked paths under
`examples/neuroprobe_eval`, including 42 notebook paths and 29 leaderboard paths,
so the initial subtree filter projects 264 tracked paths before the pre-import
privacy/license gate. The final audited baseline commit `441aad6` contains 221
subtree files plus three exact root license/notice files. Its untracked
`popt_comparability` config/script are intentionally excluded by the tracked-only
contract.

Use `/home/geeling/Projects/ieeg_project/torch_brain/examples/neuroprobe_eval/notebooks`
as a separate analysis-only source for notebook-to-YAML/output provenance, and
`/home/geeling/Projects/ieeg_project/torch_brain/examples/neuroprobe_eval/outputs/09_neurips`
as the read-only original-result source during reference capture. This second
worktree is intentionally dirty at base commit
`47a3c8900c1f6bf2aec79bb08d99c389e4036603`; do not treat its base SHA as a full
description of the notebook/YAML state. Record paths, hashes and Git tracked/
untracked status for every notebook/YAML/reference artifact actually consulted.
Do not copy its runtime, configs or scripts into the baseline.

If the clean `torch_brain-main` baseline lacks a launcher needed to reproduce a
selected parity case, consult
`/home/geeling/Projects/ieeg_project/torch_brain/examples/neuroprobe_eval/scripts`
as an analysis-only source for command structure, loop coverage and resolved
overrides. Record the path, SHA256 and tracked/untracked status of every script
consulted. Reimplement the required behavior as a small canonical iMINDBench
launcher against the imported runtime; do not mechanically copy dirty scripts,
private paths, cluster assumptions, broad experiment matrices or unrelated
model support. Record any behavioral translation in `CHANGELOG_PARITY.md`.

Run a pre-import public-release scan over the filtered source tree for secrets,
credentials, private hosts/user paths, PII, restricted data/checkpoints, large
binaries and incompatible licenses. If anything sensitive or non-redistributable
is found, stop and fix the source/scope before the baseline commit; deleting it in
a later commit would leave it in public Git history.

The pre-import audit found no credentials, checkpoints, neural data or binaries,
but it did find private user/cluster paths and a personal email in archived and
Slurm launchers. Those files are excluded from the public baseline before Git
history is created. Pseudonymous stable subject/session JSON artifacts are
explicitly retained for the public benchmark. BaRISTA is subject to its USC
research/non-profit license, so the baseline also imports exact tracked copies of
the source repository's `LICENSE.txt`, `THIRD_PARTY.md`, and
`LICENSES/BaRISTA-LICENSE.md`; commercial use of BaRISTA requires USC permission.

Use an explicit baseline-import commit so Git records the migration accurately:

1. Resolve the source file list from Git at the recorded `torch_brain` source SHA,
   not from a filesystem-wide copy. Copy every tracked file under
   `examples/neuroprobe_eval` except `outputs/**`, `notebooks/**`,
   `leaderboard/**`, `scripts/archive/**`, and audited Slurm launchers containing
   private paths/personal email, preserving bytes and relative paths exactly beneath the
   destination package directory `iMINDBench/neuroprobe_eval/`. Untracked files,
   ignored files, caches, temporary files and Git metadata must not enter the
   destination.
   `leaderboard/**` is permanently out of scope—not deferred for a later commit—
   because the leaderboard UI, submission data, validation and publishing
   workflow will live in a separate repository. iMINDBench may document the
   result JSON contract it emits, but must not vendor or depend on leaderboard
   implementation code or assets.
2. Make the first feature-branch commit contain only that untouched snapshot:
   `chore: import neuroprobe_eval baseline from <source-sha>`. Do not include this
   plan, the changelog, formatting, renames, generated files, dependency edits or
   cleanup in the baseline commit. Verify the exact relative path, Git blob OID
   and file mode for every projected source-tree entry, including executable bits
   and symlinks; verify there are no extra staged paths. The source prefix
   `examples/neuroprobe_eval/` maps to `iMINDBench/neuroprobe_eval/`. Compare the
   three root license/notice additions separately against their exact source Git
   blob IDs and modes.
3. Commit this migration plan and `CHANGELOG_PARITY.md` as the second commit,
   filling in the source and baseline commit identifiers.
4. Analyze the source paper notebooks in the original `torch_brain` checkout
   without copying the notebooks themselves. Add only the selected YAMLs and
   provenance mappings required by public launchers in later explicit commits.
5. After the baseline and migration documents are committed, make a dedicated
   namespace commit named `refactor: rename neuroprobe_eval package to
   imindbench`. Rename the package directory to `imindbench/` and update internal
   imports, Hydra module/config resolution, scripts, tests, packaging metadata,
   console entrypoints and documentation together. Do not mix algorithm,
   preprocessing or parity changes into this rename commit.
6. Perform packaging, import migration, path cleanup, dependency reduction,
   script consolidation and parity fixes in small reviewable commits. Update
   `CHANGELOG_PARITY.md` in the same commit as each behavior-affecting change.
7. After the baseline is safely recorded, reduce it to the public surface below.
   This makes deletions and modifications directly reviewable against the source.

Preserve the package-relative layout while retaining only:

- runtime: `run_eval.py`, `base_runner.py`, `utils/`, required `models/`, and
  required `preprocessors/`;
- configuration: only `conf/` entries reachable from retained release scripts;
- launchers: a small canonical runner plus publication/parity manifests; remove
  archive, backfill, rebuttal, cluster-specific and exploratory scripts;
- stable artifacts: `decodable_subject_sessions/`, the result-schema utilities,
  selected publication YAMLs moved into a non-notebook config directory, and
  documented model checkpoints;
- publication provenance: selected YAML specifications, stable artifacts, and
  reference-result mappings derived from the source paper notebooks; do not copy
  the notebooks into iMINDBench;
- packaging: `pyproject.toml`, pinned `environment.yml`, README,
  license, data-download instructions and tests.

The retained model surface is the legacy paper families—Linear/BrainBERT,
Logistic, MLP, CNN, HTNet, PopT, and DIVER—plus BaRISTA. Preserve BaRISTA's
released preprocessing for reference parity: split-local, same-recording window
concatenation for filtering; direct 1000-to-2048 Hz upsampling for BYD; native
2048 Hz for PIPPI/NeuroprobeV2; Laplacian rereferencing; global robust scaling;
and per-window/per-channel z-scoring. At a high level this remains comparable to
the model-specific HTNet and DIVER preprocessing used by the benchmark, but it
must be reported as its own model/preprocessor configuration. Retention is not a
claim that every model has a frozen or currently comparable parity case.

Do not import MVPFormer or SEEGnificant from their later release commits. During
surface reduction, explicitly inventory and remove any residual scripts,
configs, preprocessors, model registrations, dependencies and documentation
whose only purpose is MVPFormer or SEEGnificant. Verify static Hydra/config
reachability, imports and retained tests before deletion so shared functionality
used by BaRISTA or legacy models is not removed accidentally. Consolidate the
remaining launch scripts around the retained legacy models plus BaRISTA.

The relevant source tests live outside the imported subtree under
`torch_brain/tests/test_neuroprobe_eval_pipelines`. Inventory their source blob
IDs and port the retained subset in a dedicated post-baseline test commit before
deleting runtime/config paths. The required root license/notice material is
already part of the audited baseline allowlist. Maintain retained/removed file manifests with the
reason, entrypoint/config reachability and license provenance for each group;
dynamic Hydra imports mean a single runtime trace is not sufficient evidence for
deletion.

Replace `brainsets.datasets` imports with `torch_brain.datasets`, remove absolute
paths and private defaults, and make data/results/checkpoints configurable from a
single paths file. Generated outputs, caches, raw data and processed H5 files must
be ignored. Add a provenance manifest mapping every source paper notebook to its
YAML specs, result directories, launch script and stable artifacts. Use import
tracing and the smoke/parity runner to delete unreachable code only after the
copied version works. Notebook re-execution is not a release validation
requirement.

Add ignore rules immediately after the baseline for environments, caches,
raw/processed H5 data, checkpoints, Hydra scratch and candidate outputs. Keep
tracked `artifacts/parity_reference/**` on an explicit allowlist while ignoring
generated `artifacts/parity_reports/**` and `outputs/**`.

## 3. Publication parity gates

The implemented interface is the checked-in
`artifacts/parity_reference/manifest.json` plus `scripts/parity_tools.py`; there
is no experiment-running `run_parity_subset.sh`. The tool validates inputs,
constructs commands with `build-commands`, or compares caller-supplied candidate
JSON with `compare`. Both modes require caller-owned JSON checkpoint/resource
mappings; command construction additionally requires a data root and fresh output
root, while comparison requires a candidate map and fresh report directory.

The five selected records are NeuroprobeV2 Logistic/multi-STFT onset, BYD
MLP/multi-STFT global flow, PIPPI PopT/multi-STFT speech, NeuroprobeV2
BaRISTA/waveform onset, and NeuroprobeV2 PopT/multi-STFT hold-in onset. Logistic
and MLP are runnable metric/config-record cases. The two PopT cases and BaRISTA
are `NOT-COMPARABLE` because the historical results do not establish exact
checkpoint hashes; a checkable identity, config, fold, metric, or supplied known
checkpoint mismatch is still `FAIL`.

```bash
python scripts/parity_tools.py --manifest artifacts/parity_reference/manifest.json build-commands --case neuroprobev2_logistic_multistft_onset_sub1_sess1 --data-root /path/to/processed --output-root /path/to/fresh-run-root --checkpoint-map checkpoint-map.json --resource-map resource-map.json
python scripts/parity_tools.py --manifest artifacts/parity_reference/manifest.json compare --case neuroprobev2_logistic_multistft_onset_sub1_sess1 --candidate-map candidate-map.json --checkpoint-map checkpoint-map.json --resource-map resource-map.json --report-dir /path/to/fresh-report-root
```

The CLI exits zero only when every selected comparison is `PASS`; `FAIL`,
`MISSING`, and `NOT-COMPARABLE` are nonzero.

The redistributable reduced records freeze source/result URIs and hashes,
config/preprocess hashes, identity, seed, folds, and reported metrics. Historical
result counts are recorded as unavailable, and historical environment,
dataset/H5, label, and checkpoint fingerprints remain evidence gaps rather than
inferred values. The comparator consumes only checked-in immutable reference
records and verifies their hashes; it does not read a mutable private TorchBrain
checkout. Tests cover offline resolution, path containment, tampered hashes,
non-finite metrics, status precedence, CLI exits, and atomic report finalization.

Keep two separate gates:

1. **Evaluation-code parity:** run the untouched imported baseline and later
   migrated revisions from iMINDBench against the exact same immutable processed
   H5/checkpoint/config inputs, then compare to frozen original result records.
   No new run may execute from either TorchBrain source checkout.
2. **Public end-to-end reproduction:** run iMINDBench against newly generated
   `brainsets prepare` outputs. Differences may indicate preparation/data-version
   drift and must not automatically be labeled evaluation regressions. If input
   fingerprints differ and equivalence is not established, report
   `NOT-COMPARABLE`.

The frozen manifest sets absolute and relative metric tolerances to `1e-9` and
compares exact fold indices plus per-fold metrics; it does not aggregate folds or
claim hardware-repeat equivalence. Any future repeat-count or nondeterminism
policy must be set before observing new candidate scores. Reports are
machine-readable JSON plus Markdown under a caller-selected fresh directory
(conventionally `artifacts/parity_reports/<run_id>/`). Historical sample/class
counts are null, so they cannot be compared for the five frozen records.

### Notebook-to-run provenance and smoke parity

Scope notebook analysis exclusively to
`/home/geeling/Projects/ieeg_project/torch_brain/examples/neuroprobe_eval/notebooks/paper_figs/`.
Use those paper-figure notebooks only to inventory what produced the reported
analyses; other notebooks are out of scope unless a `paper_figs` notebook
directly names a stable artifact produced by one:

1. For each notebook, extract every referenced `notebooks/visualize_model/*.yaml`
   and any direct result path/filter.
2. Resolve each YAML entry to its canonical output root, dataset, regime, model,
   preprocessor, task/subset assumptions, and required checkpoint.
3. Record this mapping in a checked-in provenance manifest with one canonical
   iMINDBench launch script (or manifest-driven launcher entry) for every retained
   output family. Flag notebook inputs that cannot be traced to a reproducible
   command; do not infer or silently repair them.

The validation harness is script/result based, not notebook based. The committed
parity tool deliberately does not execute experiments: `build-commands` emits the
validated commands for selected or all manifest cases, and the caller runs any
runnable command into the required fresh output root before invoking `compare`.
During reference capture, the canonical original results under
`torch_brain/examples/neuroprobe_eval/outputs/09_neurips` are read-only; routine
comparison uses only the frozen redistributable reference artifacts.

All new baseline runs must execute the installed copied `neuroprobe_eval` package;
all post-rename candidate, parity and end-to-end runs must execute the installed
`imindbench` package. The current comparator explicitly reports
`execution_provenance_verified: false`: a `PASS` proves only metric/config-record
parity. Before reporting execution reproduction, the run workflow must separately
capture and verify imported package paths, source/environment identity, resolved
configuration, command, and data fingerprints, and fail if iMINDBench resolves
inside either TorchBrain source checkout.

Command construction fails if its output root already exists; comparison fails if
its report directory exists and finalizes JSON/Markdown atomically through a
same-parent temporary directory. There is no resume mode. Actual run orchestration
must keep candidate outputs, Hydra directories, logs and caches in the caller's
isolated root and add resolved config plus source/environment/data fingerprints
and atomic completion markers before it can claim execution provenance.

For every selected case, the candidate map links its case ID to one existing
`population_*.json`. Compare model/preprocessor/task, subject/session, regime,
seed, time-bin identity, config/preprocess hashes, folds and reported metrics.
Sample/class counts are unavailable in the historical records. Use exact
comparison for deterministic identity/hash/fold fields and the manifest tolerance
for metrics. Emit:

- `artifacts/parity_reports/<run_id>/report.json` for machine-readable status;
- `artifacts/parity_reports/<run_id>/report.md` for review;
- per-case candidate, reference, source-result, checkpoint, and resource hashes
  without embedding caller-private absolute paths;
- PASS/FAIL/MISSING/NOT-COMPARABLE status with metric deltas and reasons.

Keep `CHANGELOG_PARITY.md` as the review log for every copied-code adjustment
needed to make the smoke suite run or match. Each entry must identify the files,
reason, behavioral effect, affected smoke cases, before/after result, and whether
the change is portability-only or changes computation. Small expected changes
include import paths, configurable filesystem paths, packaging, and removal of
private defaults. Flag before proceeding with substantial parity work: algorithm,
preprocessing, split, seed, model/checkpoint, output-schema changes; broad
refactors; or debugging that affects multiple cases or materially changes metrics.
Do not tune the copied implementation against reference scores merely to force a
match.

Notebook provenance gaps remain reportable findings rather than notebook-rerun
failures. Carry forward the known Figure 4b manual-selection gap and Appendix 6
historical-rendering gap in the provenance manifest and change log.

## 4. Release sequence

1. **Complete:** Bootstrap the iMINDBench repository/feature branch; scan and verify the fixed
   `torch_brain-main` source SHA above; make its filtered untouched baseline the
   branch's first commit, then commit the plan/changelog/provenance.
2. **Partial:** Freeze licensed reference inputs/results and port the relevant
   source tests. The intended untouched imported-baseline smoke case was not run
   before simplification and is now an explicit historical evidence gap; do not
   recreate it by launching from a TorchBrain checkout. The reduced reference
   records and tests are complete, while historical data/environment/checkpoint
   fingerprints remain gaps.
3. **Partial:** Create the `imindbench` environment, package the copied runtime, then make the
   dedicated `neuroprobe_eval` → `imindbench` namespace commit and migrate public
   imports/paths. Make one Logistic evaluation-code parity case pass against
   identical immutable inputs. Packaging and namespace migration are complete;
   clean-environment installation and the Logistic run/compare gate remain.
4. **Tooling complete; execution pending:** Reduce the committed baseline using
   retained/removed manifests and tests, then add the remaining parity matrix,
   command builder, comparator and reports. The tool is intentionally not an
   experiment runner, and no candidate report is a committed acceptance result.
5. **Complete:** Trace source paper notebooks to YAMLs/original outputs and add
   canonical launch mappings without requiring notebook execution; retain the
   documented Figure 4b and Appendix 6 gaps.
6. **Partial:** All `--list` and single-recording prepare/load/idempotency gates
   pass, and full PIPPI preparation plus its full idempotent rerun pass. Full
   Neuroprobe2025 preparation and all four NeuroprobeV2 regime checks remain;
   full BYD preparation remains optional when resources permit. Treat fresh-data
   reproduction separately from evaluation-code parity.
7. **Pending:** From a fresh clone and built iMINDBench/TorchBrain artifacts—not editable
   installs—validate in `imindbench` with public dependencies and no private
   absolute paths, including one GPU case, lint/tests, two simplification/bug-risk
   reviews, and the documented smoke workflow.

### GPU handoff and push pause

After the clean CPU gates and at least one runnable parity comparison are
reviewed, prepare a concise GPU handoff containing the exact iMINDBench and
TorchBrain SHAs, environment/export and CUDA diagnostics, the runnable
checkpoint-free BYD MLP manifest case (unless the user selects another bounded
case), caller-owned data and checkpoint/resource mapping hashes, generated command, fresh
output/report roots, and expected result/report files. Do not launch an unreviewed
broad experiment matrix as the acceptance gate.

Before pausing, commit every coherent CPU phase and, under the user's standing
authorization for this GPU handoff, push both feature branches so another agent
or machine can resume from the recorded SHAs. Do not merge branches or publish
release artifacts. If credentials prevent the push, record the exact
local SHAs, remotes, branch names, failed authentication method, and push commands
in the handoff rather than losing or rewriting local history. Then pause after
handing off the single GPU case. A metric/config `PASS` alone is not release
acceptance because the current comparator does not verify execution provenance.
