# Frozen parity references

`manifest.json` defines five paper-figure parity cases and hashes every reduced
reference record in `cases/`. The `original-results://` URIs are stable source
identifiers, not local filesystem paths.

Logistic and MLP can pass metric/config-record comparison. PopT and BaRISTA are
`NOT-COMPARABLE` only when every checkable identity, config hash, fold, and metric
matches but the historical checkpoint hash remains unknown. Any checkable
mismatch is `FAIL`, including for those cases. Supplying a checkpoint records its
hash but does not close the historical provenance gap.

`scripts/parity_tools.py` only constructs commands and compares existing JSONs.
It never launches an experiment. Both run output roots and report directories
must be fresh, and all data, output, checkpoint, and auxiliary resource paths are
provided by the caller rather than stored here.

`PASS` means only that the candidate JSON matches the frozen metric/config record
within the declared tolerance. It does not prove which code, data, imports,
resolved configuration, or command produced the candidate. Reports state
`execution_provenance_verified: false`; do not interpret them as execution
reproduction evidence.

The comparison CLI exits zero only if every selected case is `PASS`. It exits
nonzero for `FAIL`, `MISSING`, and `NOT-COMPARABLE`. Candidate JSON bytes,
supplied checkpoint files, and auxiliary resources are fingerprinted in reports;
directory resource hashes cover sorted relative file names and file contents.
