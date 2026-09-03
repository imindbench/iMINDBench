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

`PARITY_MATRIX.md` summarizes those directories by dataset/model/preprocessor
family, run era, data-layer lineage, and required backward-parity adjustments.
It also gives a clean-checkout replication procedure without binding the
documentation to one machine's absolute paths.

BaRISTA is genuinely used: the Appendix 2 and Table 1 waveform YAMLs include its
NeuroprobeV2, BYD, and Pippi result families. However, the only consulted BRANT/
BaRISTA-style script does not reproduce the exact `barista_jun9` output roots,
preprocessor, or checkpoint mapping. This is a provenance gap, not evidence that
BaRISTA should be dropped.

## Runtime profiles represented in paper figures

The paper notebooks combine results produced under two runtime profiles. These
profiles belong to result families, not entire figures: a single notebook may
read both historical and newer families. The resolved `.hydra/config.yaml` beside
each result remains authoritative.

- **Historical profile:** `runner.num_workers=4`, `runner.pin_memory=true`,
  `runner.persistent_workers=true`, `runner.prefetch_factor=2`, and
  `runtime.preprocess_torch_num_threads=6`. This profile covers the original
  NeuroprobeV2, BYD, and Pippi flagship multi-STFT Logistic/MLP/CNN/PopT results;
  NeuroprobeV2 half/quarter/eighth/sixteenth sample-efficiency results; the
  original STFT sweep families used by Figure 4 and Appendix 1; `barista_jun9`;
  and the older NeuroprobeV2 multisource family. These results feed the flagship
  panels in Appendix 2 and Table 1, the sample-efficiency panels in Figure 3 and
  Appendix 4, and parts of the task/coverage and preprocessing figures.
- **Newer profile:** `runner.num_workers=0`, `runner.pin_memory=false`,
  `runner.persistent_workers=false`, `runner.prefetch_factor=2`, and
  `runtime.preprocess_torch_num_threads=4`. This profile covers `new_baselines`,
  `rebuttal_preprocessor_baselines`, `rebuttal_holdin_multisource_exps`,
  `diver_outputs_longcontext`, and the dataset-specific
  `{dataset}_stft_{sampling_rate}` baseline roots. These results feed the newer
  waveform/baseline panels in Appendix 1, Appendix 2, Figure 4, and Table 1, plus
  the rebuttal hold-in/multisource panels used by Figure 3 and Appendix 5.

The `kelesbyd2024_tercile` root is mostly historical. Its
`logistic_laplacian_stft_1000Hz` and
`logistic_laplacian_multi_stft_1000Hz_zscore` families use the newer profile;
the other active YAML paths sampled from that root use the historical profile.
Across the 102 unique result paths resolved from the 23 active visualization
YAMLs, representative resolved configs were available for 101 paths: 57 used the
historical profile and 44 used the newer profile. This is a path-level provenance
inventory, not a weighting by the number of subject/task result files.

### Historical window-indexing semantics

The runtime profile was not the only historical condition needed for exact
provider parity. The April 2026 BYD and Pippi runs used
`temporaldata==0.1.1`, whose lazy regular-time-series slicing converted both
window boundaries with `floor`. The migrated TorchBrain data implementation
snaps boundaries to the sampling grid and uses `ceil`; for non-grid-aligned
trial boundaries this shifts each otherwise equal one-second window by one
sample. Direct comparison showed `historical[1:] == migrated[:-1]`, and a
read-only diagnostic restoring the legacy floor conversion reproduced every
historical BYD and Pippi Logistic metric exactly in both folds.

The ceil-based helper was introduced upstream in TemporalData commit
`ad3e9850b5103bb8cc4eca421431b65658563fcf` on 2026-03-16, after the 0.1.1
release used by the paper environment. It reached TemporalData v0.1.2 on
2026-05-29 and was carried into TorchBrain by the TemporalData/Brainsets merge
`6069a76c84bcbc96eb9253d1263e032046853e8f` (authored 2026-05-29, committed
2026-06-10). iMINDBench therefore inherited this semantic change during the
migration; it did not originate it. This finding is separate from the loader
and preprocessing-thread profile that restored NeuroprobeV2 MLP parity.

The same legacy-slice diagnostic was also run for the coordinate-aware PopT-v2
model on BYD global flow and Pippi speech. Both used the exact 1M-step
main-results checkpoint (SHA256
`cf4e835d5309559d468b2f1ebd9b76882398c30bedae6c0c8bc6fdb4c506b52f`),
passed each sample's LIP coordinates through
`MultiSubjBrainPositionalEncoding`, and reproduced every historical metric
exactly in both folds. The historical resolved configs identify the same
checkpoint path, closing the checkpoint-identity gap for the Pippi PopT parity
case. Categorical `brain_area_key` is separate metadata; PopT consumes numerical
channel coordinates.

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

The later `neuroprobev2_mlp_multistft_onset_sub1_sess1` record supplements this
original five-case provenance subset as the development GPU parity case. Its
migrated run used the historical paper profile (four persistent workers, pinned
memory, prefetch factor two, and six preprocessing threads) and matched every
frozen fold metric exactly on newly prepared public data. The deterministic
HTNet record remains a runnable but unexecuted optional reference.

Two later provider-specific Logistic records supplement the original subset:
BYD multi-STFT/global-flow subject 41/session 1 and PIPPI high-cov
multi-STFT/speech subject 1/session 1. Both are referenced by the paper's
flagship STFT YAMLs and retain their adjacent historical resolved configs and
run-log hashes in `consulted_artifacts.tsv`.

## Gaps to carry forward

- Figure 4b remains a manual four-panel selection rather than a notebook default.
- Appendix 6 does not reproduce the historical paper rendering recipe.
- Exact launch commands are missing for BaRISTA, HTNet 500 Hz, preprocessing
  baselines, and NeuroprobeV2 sample-efficiency roots.
- Current dataset launch scripts are useful command-structure evidence but their
  active model arrays do not reproduce every historical YAML entry.

No private absolute path is recorded in the manifest or report.
