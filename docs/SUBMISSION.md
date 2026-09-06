# Submission source packaging

Build the source archive with:

```bash
python scripts/build_submission.py --output /path/to/imindbench-submission.tar.gz
```

The output must not already exist. `config/submission_files.json` is the explicit
reviewed file allowlist. The exporter needs no Git, rejects symlinks and known
personal-path patterns, and normalizes archive owner/time metadata. It includes
file hashes, code, configs, tests and required license notices. It excludes Git
history, local paths profiles, model weights, datasets,
outputs, build remnants and personal migration records.

To include the reviewed TorchBrain dependency, add
`--torch-brain-wheel /path/to/torch_brain-<version>-py3-none-any.whl`.
The exporter checks its package/source identity and the exact reviewed SHA256,
scans textual wheel contents,
bundles it under `vendor/`, and points the exported quickstart at the local wheel.
`config/torch_brain_dependency.json` pins the exact dependency artifact. This
avoids requiring a Git installation step for TorchBrain. Normal dependencies
still require network access unless already available;
the optional warmup scheduler also retains a Git installation requirement.

Extract into a fresh directory and follow the quickstart. TorchBrain is a
separate required dependency; the documented immutable public commit remains
the tested source pin. If linking that benchmark-specific dependency could
identify the submission, provide its reviewed artifact separately and document
its hash and installation in the submission. The exporter can bundle an existing
reviewed TorchBrain wheel; it does not
build that wheel or remove required upstream attribution/source fingerprints.

Before submission, inspect the final archive and rendered documentation for
identifying links, scientific names and third-party metadata. A known-pattern
scan cannot establish complete anonymity. Preserve upstream attribution and
license notices. Checkpoints and dataset access requirements must be documented
separately; neither is supplied by this source archive.

The canonical root README is included as package metadata and rewritten to use
the local TorchBrain wheel when bundled. Only allowlisted release files enter
the archive; development history and private verification records are excluded.
