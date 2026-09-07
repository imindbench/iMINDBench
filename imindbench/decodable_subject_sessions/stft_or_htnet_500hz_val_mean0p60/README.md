# Decodable subject/session units

These provider manifests use the `stft_or_htnet_500hz_val_mean0p60` rule:
a subject/session is eligible when the maximum of its best STFT validation
mean ROC AUC and its HTNet 500 Hz validation mean ROC AUC is strictly greater
than 0.60. The JSON files preserve the selection metadata and task-specific
subject/session lists.

Files are named by provider:

- `kelesbyd2024.json`
- `berezutskayapippi2022.json`
- `neuroprobev2.json`

The `within-dataset` and `multi-dataset` cases in `scripts/run_experiments.sh` (relative
to the project root) select this named rule.
`imindbench-grid` filters evaluation targets before creating run directories and
passes the same directory to training through
`paths.decodable_subject_sessions_dir`. With
`dataset.train_decodable_subject_sessions_only=true`, the adapter filters each
provider's training recordings using the manifest for `dataset.task`.

Subject/session identifiers have the form `sub42_sess1`. For multisource
training, decodable filtering requires each training source to use
`hold-in-session`; each contributes eligible recordings for that provider.

Use `--decodable-dir /path/to/manifests` with the grid launcher to select a
caller-owned population, or set `paths.decodable_subject_sessions_dir` explicitly
for a direct evaluation. The directory must contain the required provider JSONs.
The parent directory is a container for named rules, not a manifest directory.
