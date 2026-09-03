# Frozen parity references

`manifest.json` defines nine paper-figure parity cases and hashes every reduced
reference record in `cases/`. The `original-results://` URIs are stable source
identifiers, not local filesystem paths.

The three Logistic cases, BYD MLP, NeuroprobeV2 MLP, and deterministic HTNet are
runnable metric/config-record comparisons. NeuroprobeV2 Logistic passes; fresh
BYD and PIPPI Logistic executions match identity/configuration but fail the
strict metric contract with small metric drift. The NeuroprobeV2 MLP case is the
completed development GPU parity evidence and records the historical paper-run
worker/thread profile. Release acceptance also passed from the pinned,
non-editable built artifacts; the comparator's `PASS` remains limited to the
metric/config record.
HTNet remains a runnable, unexecuted reference from the Figure 4 preprocessing
baseline family. PopT and BaRISTA are
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
