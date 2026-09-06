# Historical reproduction coverage

Existing parity experiments establish the migration's core numerical
compatibility. The release cleanup preserves this evidence without expanding
or rerunning the experiment matrix.

| Path | Existing evidence | Practical limit |
| --- | --- | --- |
| NeuroprobeV2 Logistic | Exact selected-case metric/config parity | Not a full sweep rerun |
| NeuroprobeV2 MLP | Exact selected-case parity, including installed-wheel execution | Historical worker/thread profile required |
| BYD/PIPPI Logistic and PopT | Exact selected diagnostics using historical windows | Clean CLI does not yet expose the diagnostic legacy-floor policy |
| Other historical families | Recipes and reference results traced to varying degrees | Some checkpoint hashes/launch mappings remain unknown; BYD MLP CLI drift remains documented |

Historical BYD/PIPPI data used floor-based temporal window boundaries. Public
TorchBrain uses its newer slicing semantics. PopT coordinate-index policy is
independent of temporal slicing. Do not claim that the default CLI reproduces
all historical numbers exactly.

Checkpoint identity, population, split, model/preprocessor configuration and
runtime profile must match before attributing a difference to implementation.
The comparator's `PASS` describes metric/config-record parity; execution
provenance is separate evidence.

The full maintainer provenance inventory and family matrix remain in the source
repository. The anonymous export includes frozen reduced reference records and
comparison tooling; it excludes machine-specific migration logs. A future
integration of legacy slicing needs focused boundary tests and one existing
provider case through the supported CLI, not a new full experiment matrix.
