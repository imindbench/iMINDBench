# Reproducing experiments

Start from the [paper coverage catalog](PAPER_COVERAGE.md) and select the matching
recipe, dataset, tasks and target population. Recipes specify experimental
settings; they do not bundle data, checkpoints or paper result files.

For meaningful comparisons, retain the exact command and resolved configuration,
dataset version and preparation settings, population manifest, random seed,
package versions and checkpoint hash. Record CPU/GPU details for torch models.
Use an external output directory for each configuration. GPU training may vary
across devices and software versions even with a fixed seed.

## Known limits

Earlier BYD/PIPPI experiments used floor-based temporal window boundaries.
Public TorchBrain uses newer slicing semantics; this release does not expose
a legacy-floor option. PopT coordinate indexing is a separate setting and does
not control temporal slicing. Default commands therefore do not guarantee exact
numerical agreement with every earlier result.

Some checkpoint identities and retrieval instructions remain incomplete; see
[RESOURCES.md](RESOURCES.md). Use the required checkpoint rather than substituting
arbitrary weights from the same model family.

The task/target catalog does not reconstruct every historical per-run setting
or population. Full paper-figure rendering is not included. Consult
[PAPER_COVERAGE.md](PAPER_COVERAGE.md) for the scope of each workflow.
