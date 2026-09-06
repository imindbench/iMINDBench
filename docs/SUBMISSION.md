# Submission source packaging

Build the source archive with:

```bash
python scripts/build_submission.py --output /path/to/imindbench-submission.tar.gz
```

The output must not already exist. `config/submission_files.json` is the explicit
reviewed file allowlist. The exporter needs no Git, rejects symlinks and known
personal-path patterns, and normalizes archive owner/time metadata. It includes
file hashes, code, configs, tests, reduced parity references and required license
notices. It excludes Git history, local paths profiles, model weights, datasets,
outputs, build remnants and personal migration records.

Extract into a fresh directory and follow the quickstart. TorchBrain is a
separate required dependency; the documented immutable public commit remains
the tested source pin. If linking that benchmark-specific dependency could
identify the submission, provide its reviewed artifact separately and document
its hash and installation in the submission. This exporter packages iMINDBench;
it does not create or anonymize TorchBrain artifacts.

Before submission, inspect the final archive and rendered documentation for
identifying links, scientific names and third-party metadata. A known-pattern
scan cannot establish complete anonymity. Preserve upstream attribution and
license notices. Checkpoints and dataset access requirements must be documented
separately; neither is supplied by this source archive.
