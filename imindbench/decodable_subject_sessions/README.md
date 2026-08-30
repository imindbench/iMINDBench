# Decodable Subject/Session Manifests

This directory contains provider-specific manifests used by
`dataset.train_decodable_subject_sessions_only=true`.

Each file is named:

```text
<provider>.json
```

For each `dataset.task`, the manifest lists subject/session pairs such as
`sub42_sess1`. Train splits are filtered by matching provider recording IDs to
these pairs.

Launch scripts also use these manifests as a target-run precheck: a subject/session
not listed for the task is skipped before output directories are created or eval is
launched.

For multisource training, decodable filtering is supported only with
`train_sources[].regime: hold-in-session`; each source then contributes all
eligible task-decodable subject/sessions for that provider.

Current manifests are provided for:

- `kelesbyd2024`
- `berezutskayapippi2022`
- `neuroprobev2`
