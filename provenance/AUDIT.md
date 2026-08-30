# Paper-figure provenance audit

Scope was limited to `notebooks/paper_figs`; notebooks and experiments were not
executed. Source artifacts were read from the dirty TorchBrain checkout at HEAD
`47a3c8900c1f6bf2aec79bb08d99c389e4036603`. The untracked source changed during
the audit: the README and Appendix 5 notebook changed and a separate Figure 3
beyond-within-session notebook appeared. The manifest records both observations
and pins the final state. Every consulted file is pinned by
SHA256 and Git status in `consulted_artifacts.tsv`; paths in checked-in outputs
are source-relative URIs.

## Result

All 14 scoped notebooks are mapped in `paper_figs_provenance.json`. They actively
reference 23 `visualize_model` YAMLs plus direct config/data inputs. All 154 active
YAML entries resolve to 102 unique result directories under the read-only
`09_neurips` tree; every directory was present during this audit. The notebooks
contain no MVPFormer or SEEGnificant reference.

BaRISTA is genuinely used: the Appendix 2 and Table 1 waveform YAMLs include its
NeuroprobeV2, BYD, and Pippi result families. However, the only consulted BRANT/
BaRISTA-style script does not reproduce the exact `barista_jun9` output roots,
preprocessor, or checkpoint mapping. This is a provenance gap, not evidence that
BaRISTA should be dropped.

## Recommended parity subset

The five selected cases are immutable candidates, not copied reference fixtures:

1. NeuroprobeV2 Logistic + multi-STFT, onset, subject 1/session 1.
2. BYD MLP + multi-STFT, global optical flow, subject 41/session 1.
3. Pippi PopT + multi-STFT, speech, subject 1/session 1, high-cov.
4. NeuroprobeV2 BaRISTA waveform, onset, subject 1/session 1.
5. NeuroprobeV2 PopT + multi-STFT, hold-in-session onset, subject 1/session 1.

Each is in the paper's `stft_or_htnet_500hz_val_mean0p60` decodable population,
has folds 0 and 1, and has an exact source JSON hash in the manifest. Together
they cover all three datasets, retained classical/torch/checkpoint paths,
language and movie-derived tasks, within/hold-in routing, and BaRISTA's released preprocessing. Before
running them, translate the dirty scripts into one public, manifest-driven
launcher and recover/freeze BaRISTA's exact checkpoint provenance.

## Gaps to carry forward

- Figure 4b remains a manual four-panel selection rather than a notebook default.
- Appendix 6 does not reproduce the historical paper rendering recipe.
- Exact launch commands are missing for BaRISTA, HTNet 500 Hz, preprocessing
  baselines, and NeuroprobeV2 sample-efficiency roots.
- Current dataset launch scripts are useful command-structure evidence but their
  active model arrays do not reproduce every historical YAML entry.

No private absolute path is recorded in the manifest or report.
