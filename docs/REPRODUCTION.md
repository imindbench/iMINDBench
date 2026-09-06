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

## Paper-reference parity tools

[`scripts/parity_tools.py`](../scripts/parity_tools.py) is an offline tool. It
constructs commands but never launches them, and it compares existing result
JSONs against the nine records in
[`artifacts/parity_reference/`](../artifacts/parity_reference/). Inspect its
validated interfaces with:

```bash
python scripts/parity_tools.py build-commands --help
python scripts/parity_tools.py compare --help
```

Both modes require caller-owned JSON mappings. `checkpoint-map.json` maps the
logical keys `popt_checkpoint` and/or `barista_checkpoint` to files when a
selected case needs them. `resource-map.json` maps auxiliary resource keys; an
empty object is valid when the selected case needs none. Comparison additionally
uses `candidate-map.json`, mapping case IDs to existing `population_*.json`
files. Run and report output roots must not already exist.
For the checkpoint-free NeuroprobeV2 MLP example below, both `checkpoint-map.json` and
`resource-map.json` may contain the single JSON object `{}`.

Example GPU command construction for the checkpoint-free MLP case:

```bash
python scripts/parity_tools.py \
  --manifest artifacts/parity_reference/manifest.json \
  build-commands \
  --case neuroprobev2_mlp_multistft_onset_sub1_sess1 \
  --data-root /path/to/processed \
  --output-root /path/to/fresh-run-root \
  --checkpoint-map checkpoint-map.json \
  --resource-map resource-map.json \
  --device cuda:0
```

Example comparison:

```bash
python scripts/parity_tools.py \
  --manifest artifacts/parity_reference/manifest.json \
  compare \
  --case neuroprobev2_mlp_multistft_onset_sub1_sess1 \
  --candidate-map candidate-map.json \
  --checkpoint-map checkpoint-map.json \
  --resource-map resource-map.json \
  --report-dir /path/to/fresh-report-root
```

`PASS` means metric/config-record parity only; it does not prove execution
provenance. `FAIL` and `MISSING` are nonzero exits. The PIPPI PopT-v2 reference
is checkpoint-bound and runnable; the remaining NeuroprobeV2 PopT and BaRISTA
references remain `NOT-COMPARABLE`—also a nonzero exit—when every checkable
field matches but the exact historical checkpoint hash is unknown. See the
[`parity reference README`](../artifacts/parity_reference/README.md) for the
precise status contract.
