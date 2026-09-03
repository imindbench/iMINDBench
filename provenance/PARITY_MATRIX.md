# Paper-output parity matrix

This is the human-readable companion to `paper_figs_provenance.json`. It groups
the 102 unique result directories referenced by the 14 scoped paper notebooks
and their 23 active visualization YAMLs. It is exhaustive at the output-family
level; the JSON inventory remains authoritative for the individual directories.

Dates below come from result-record timestamps, not YAML directory names. The
historical runs did not capture complete environment manifests, so package
versions are reported only where there is direct evidence.

## Backward-parity adjustments

| Mark | Meaning |
|---|---|
| ✓ | Required for matching this historical family |
| ✗ | Not required for this family |
| ? | Historical evidence is incomplete; inspect the result-adjacent resolved config or recover the missing artifact |
| — | Not applicable |

The adjustment columns mean:

- **H — historical runtime:** `num_workers=4`, pinned memory, persistent workers,
  prefetch factor 2, and six preprocessing threads. The newer profile is
  0/false/false/2/4.
- **F — legacy floor windows:** use the pre-merge TemporalData behavior that
  converts both time-window boundaries with `floor`, rather than the integrated
  data layer's grid-snap/`ceil` behavior.
- **C — legacy PopT coordinates:** use the historical numerical LIP-coordinate
  indexing policy (`runner.coord_index_policy=legacy_truncation`). This is
  independent of F; it does not control time slicing.
- **K — exact resource:** verify the historical checkpoint or other learned
  resource by SHA256 before running. If its historical hash is unknown, strict
  execution parity is `NOT-COMPARABLE`.

## Matrix

The inventory's ten output-family IDs map to the rows below as follows:

| Inventory family | Matrix coverage |
|---|---|
| `stft_sweeps` | Dataset-specific Logistic sweep rows |
| `htnet_500hz_waveform` | July `new_baselines` and August HTNet rows |
| `flagship_multistft` | Original-root Logistic/MLP/CNN/PopT-v2 rows |
| `flagship_waveform` | Original HTNet, BaRISTA, DIVER-1, and July `new_baselines` rows |
| `logistic_stft_variants` | Dataset-specific Logistic rows |
| `neuroprobev2_sample_efficiency` | NeuroprobeV2 sample-efficiency and older hold-in/multisource rows |
| `holdin_multisource` | Older NeuroprobeV2 and July rebuttal PopT-v2 rows |
| `preprocessor_baselines` | Original STFT/raw-waveform and August rebuttal rows |
| `brainbert_stft` | BrainBERT linear and July dataset-specific single-STFT rows |
| `processed_data_figure` | Final non-evaluation row |

| Dataset | Model / preprocessor / result family | Paper use | Approx. run | BrainSets/data-layer evidence | TemporalData/window evidence | H | F | C | K | Current parity position |
|---|---|---|---:|---|---|:---:|:---:|:---:|:---:|---|
| NeuroprobeV2 | Logistic; `neuroprobev2_stft_sweep_with_val` and original-root single/multi/high STFT | Appx. 1/2/3; Fig. 4; Table 1 | Apr 2026 | Standalone historical BrainSets state; exact revision not recorded | 0.1.1-era; boundaries used here are grid aligned | ✓ | ✗ | ✗ | ✗ | Multi-STFT representative is exact; the whole sweep was not rerun |
| PIPPI | Logistic; `berezutskayapippi2022_stft_sweep_with_nonword_tercile` and original-root single/multi/high STFT | Appx. 1/2/3; Fig. 4; Table 1 | Apr 2026 | Standalone historical BrainSets state; exact revision not recorded | `temporaldata==0.1.1` floor semantics verified | ✓ | ✓ | ✗ | ✗ | Multi-STFT representative is exact with F; other STFT variants inherit the data-path requirement but remain untested |
| BYD | Logistic; `kelesbyd2024_stft_sweep_with_nonword_tercile` plus original-root multi/high STFT | Appx. 1/2/3; Fig. 4; Table 1 | Apr 2026 | Standalone historical BrainSets state; exact revision not recorded | `temporaldata==0.1.1` floor semantics verified | ✓ | ✓ | ✗ | ✗ | Multi-STFT representative is exact with F; other STFT variants inherit the data-path requirement but remain untested |
| BYD | Logistic; original-root single-STFT and multi-STFT-zscore exceptions | Fig. 4 | Apr/Aug 2026 records | Integrated/newer-profile result family; exact run commit not recorded | Result-adjacent configs are authoritative for the mixed-date directory | ✗ | ✗ | ✗ | ✗ | These two paths are explicit exceptions to the otherwise historical `kelesbyd2024_tercile` root |
| NeuroprobeV2 | MLP/CNN; original-root single-STFT, multi-STFT, and early z-score baselines | Appx. 2; Fig. 4; Table 1 | Apr–May 2026 | Standalone historical BrainSets state; exact revision not recorded | 0.1.1-era; selected windows are grid aligned | ✓ | ✗ | ✗ | ✗ | MLP multi-STFT is exact on new data; CNN and other preprocessors were not rerun |
| BYD, PIPPI | MLP/CNN; original-root single-STFT, multi-STFT, and early z-score baselines | Appx. 2; Fig. 4; Table 1 | Apr 2026; some BYD z-score records Aug | Standalone historical profile for the active families; exact BrainSets revision not recorded | `temporaldata==0.1.1` floor semantics applies to the historical provider path | ✓ | ✓ | ✗ | ✗ | Logistic and PopT prove F; inspect mixed-date BYD configs and rerun MLP/CNN metrics with F |
| NeuroprobeV2 | PopT-v2 + multi-STFT, flagship within-session | Appx. 2; Table 1 | Apr 2026 | Standalone historical BrainSets state; exact revision not recorded | 0.1.1-era; selected windows are grid aligned | ✓ | ✗ | ✓ | ✓ | Recipe traced; this exact family was not selected for execution acceptance |
| BYD, PIPPI | PopT-v2 + multi-STFT, flagship within-session | Appx. 2; Table 1 | Apr 2026 | Standalone historical BrainSets state; exact revision not recorded | `temporaldata==0.1.1` floor semantics verified | ✓ | ✓ | ✓ | ✓ | Exact in both folds with the main 1M-step checkpoint (`cf4e835d…`) and numerical LIP coordinates |
| NeuroprobeV2 | Logistic/MLP/CNN + multi-STFT, full through sixteenth sample efficiency | Fig. 3; Appx. 4 | Apr 2026 | Standalone historical BrainSets state; exact revision not recorded | 0.1.1-era; selected windows are grid aligned | ✓ | ✗ | ✗ | ✗ | Fraction/subset routing must match; not all fractions were rerun |
| NeuroprobeV2 | PopT-v2 + multi-STFT, full through sixteenth sample efficiency | Fig. 3; Appx. 4 | Apr 2026 | Standalone historical BrainSets state; exact revision not recorded | 0.1.1-era; selected windows are grid aligned | ✓ | ✗ | ✓ | ✓ | Fraction/subset routing and the exact checkpoint must match; not all fractions were rerun |
| NeuroprobeV2 | PopT-v2 + multi-STFT, older hold-in/multisource roots | Fig. 3; Appx. 4/5 | Apr 2026 | Standalone historical BrainSets state; exact revision not recorded | 0.1.1-era; selected windows are grid aligned | ✓ | ✗ | ✓ | ? | Not selected for formal comparison; checkpoint and exact launcher provenance must be recovered |
| NeuroprobeV2 | HTNet + original sampling-rate raw-waveform roots | Appx. 1; Fig. 4 | Apr 2026 | Standalone historical BrainSets state; exact revision not recorded | 0.1.1-era; selected windows are grid aligned | ✓ | ✗ | ✗ | ✗ | Exact historical launcher is missing; this is separate from the runnable August HTNet reference |
| BYD, PIPPI | HTNet + original sampling-rate raw-waveform roots | Appx. 1; Fig. 4 | Apr 2026 | Standalone historical BrainSets state; exact revision not recorded | Non-grid boundaries make legacy slicing relevant, but this waveform path was not tested | ✓ | ? | ✗ | ✗ | Exact launcher is also missing |
| NeuroprobeV2 | BrainBERT linear baseline + BrainBERT STFT | Table 1 | Apr–May 2026 | Standalone/transition-era state; exact revision not recorded | Selected windows are grid aligned | ✓ | ✗ | ✗ | ✓ | Exact historical checkpoint hash/launcher mapping must be recovered |
| BYD, PIPPI | BrainBERT linear baseline + BrainBERT STFT | Table 1 | Apr–May 2026 | Standalone/transition-era state; exact revision not recorded | Non-grid boundaries in the floor-era data path | ✓ | ✓ | ✗ | ✓ | Exact historical checkpoint hash/launcher mapping must also be recovered |
| All three | BaRISTA + session waveform/z-score (`barista_jun9`) | Appx. 2; Table 1 | Jun 2026 | Transition-era state around the BrainSets merge; exact revision not recorded | Exact data-package state was not captured | ✓ | ? | ✗ | ✓ | Active paper input, but launcher/checkpoint provenance is incomplete; `NOT-COMPARABLE` |
| All three | DIVER-1 + long-context waveform | Appx. 2; Table 1 | Jun 2026 | Transition/integrated data-layer state; exact run commit not recorded | No legacy-floor requirement is evidenced | ✗ | ✗ | ✗ | ✓ | Newer runtime profile; exact resource and resolved config are required |
| All three | Logistic/MLP/CNN + dataset-specific single-STFT roots | Table 1 | Jul 2026 | Integrated TorchBrain data layer; exact run commit not recorded | Grid-snap/`ceil` lineage | ✗ | ✗ | ✗ | ✗ | Match the newer runtime profile and result-adjacent config |
| All three | Linear/Logistic/MLP/CNN/HTNet + `new_baselines` long-context waveform | Appx. 1/2; Table 1 | Jul 2026 | Integrated TorchBrain data layer; exact run commit not recorded | Grid-snap/`ceil` lineage | ✗ | ✗ | ✗ | ✗ | Match the newer runtime profile and model-specific waveform transform |
| All three | PopT-v2 + multi-STFT, rebuttal hold-in/multisource | Fig. 3; Appx. 5 | Jul 2026 | Integrated TorchBrain data layer; exact run commit not recorded | Grid-snap/`ceil` lineage | ✗ | ✗ | ✓ | ✓ | Match subset/regime/source pool; freeze the checkpoint. The selected Neuroprobe case is still `NOT-COMPARABLE` because its historical checkpoint hash is unknown |
| All three | HTNet + rebuttal waveform/preprocessor variants | Fig. 4 | Aug 2026 | Integrated TorchBrain data layer; exact run commit not recorded | Grid-snap/`ceil` lineage | ✗ | ✗ | ✗ | ✗ | Match the newer profile and exact transform; the deterministic Neuroprobe reference is runnable but unexecuted |
| All three | Mixed preprocessing-baseline YAML entries | Fig. 4 | Apr–Aug 2026 | Mixed historical and integrated families | Mixed; determined per result, not per figure | ? | ? | ? | ? | Never assign one profile to the entire YAML/figure; inspect each result's `.hydra/config.yaml` |
| All three | Figure 2 data summaries and Appendix 6 input rendering | Fig. 2; Appx. 6 | 2026 paper build | Processed-data inputs, not evaluation results | Not an evaluation parity target | — | — | — | — | Reproduce data inventory separately; Appendix 6's historical render recipe remains incomplete |

The BrainSets/TemporalData merge is TorchBrain commit
`6069a76c84bcbc96eb9253d1263e032046853e8f` (authored 2026-05-29,
committed 2026-06-10). The `ceil` helper originated in TemporalData commit
`ad3e9850b5103bb8cc4eca421431b65658563fcf` and shipped in v0.1.2. These dates
bound the lineage; they do not substitute for an absent historical environment
manifest.

In particular, no exact historical BrainSets package version or commit was
captured. `temporaldata==0.1.1` is the one exact package-version finding for the
April BYD/PIPPI path. For the later rows, “integrated” names a TorchBrain-owned
data implementation, not a separately versioned BrainSets/TemporalData install.

## Replication procedure

Work from the iMINDBench checkout and refer to the public TorchBrain checkout as
the sibling `../torch_brain-public`. Do not write into the historical
`09_neurips` results. Use a new caller-owned data/run directory for every attempt.

1. Find the row/family in `paper_figs_provenance.json`, then open the named active
   visualization YAML. Resolve the precise result JSON and its adjacent
   `.hydra/config.yaml`; the latter overrides this summary table.
2. Pin the reference JSON, resolved config, run log, processed H5 files, and every
   checkpoint/resource by SHA256. Record the dataset variant, subset, regime,
   task, subject/session, folds, seed, and decodable-population file.
3. Apply H only when marked: set `runner.num_workers=4`,
   `runner.pin_memory=true`, `runner.persistent_workers=true`,
   `runner.prefetch_factor=2`, and
   `runtime.preprocess_torch_num_threads=6`. Otherwise use the newer 0/false/
   false/2/4 profile.
4. Apply F only when marked. As of this matrix, the clean iMINDBench CLI does not
   expose a legacy temporal-window policy. Exact BYD/PIPPI evidence was produced
   by a read-only diagnostic. A clean agent must either run the frozen historical
   `temporaldata==0.1.1` path or first implement a scoped, tested iMINDBench
   `legacy_floor` option. Do not change TorchBrain globally, and do not confuse
   `runner.coord_index_policy` with temporal slicing.
5. For PopT, apply C, select the exact coordinate profile, verify finite numerical
   coordinates and retained channel order, and apply K. The verified main-results
   checkpoint SHA256 is
   `cf4e835d5309559d468b2f1ebd9b76882398c30bedae6c0c8bc6fdb4c506b52f`.
6. Use `scripts/parity_tools.py build-commands` with a fresh output root and
   caller-owned checkpoint/resource maps. Review the emitted command, add only
   the row's required overrides, and execute it from the installed package.
7. Capture the exact command, result, Hydra overrides/resolved config, log,
   imported package paths/commits or wheel hashes, `pip freeze`, GPU/driver
   details for torch models, and input/resource hashes. GPU paper runs were not
   deterministic, so exact cross-CUDA repetition is not guaranteed even with
   matched inputs and settings.
8. Create a candidate map and run `scripts/parity_tools.py compare` into a fresh
   report directory. A metric `PASS` is not execution-provenance proof; report
   both conclusions separately.

Generic, non-overwriting command templates are kept in
`../coding_plans/code_notes.md`. Case IDs and currently frozen references are in
`artifacts/parity_reference/manifest.json`.
