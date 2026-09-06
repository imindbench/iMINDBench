# Data and pretrained resources

Prepare recordings with the public TorchBrain `brainsets` CLI, following the
[quickstart](../imindbench/README.md). NeuroprobeV2 shares Neuroprobe2025 prepared
artifacts. Keep raw data, prepared data, checkpoints, caches and outputs outside
the installation and use absolute paths in your external paths config.

| Model or preprocessing path | Required resource settings |
| --- | --- |
| Logistic, MLP, CNN, HTNet | No pretrained checkpoint |
| BrainBERT encoder + linear readout | `paths.brainbert_checkpoint` |
| PopT | `paths.popt_checkpoint`; numerical channel coordinates from the provider |
| BaRISTA | `paths.barista_checkpoint`; appropriate coordinate/Destrieux metadata; compatible xformers runtime |
| DIVER | `model.upstream_ckpt` and writable `model.model_dir`; select the matching coordinate and waveform config |

No pretrained checkpoints are bundled. Use the exact resource required by your
experiment and record its SHA256. A checkpoint with the same model family name
is not automatically comparable to the historical paper run. The verified
main-results PopT checkpoint hash is
`cf4e835d5309559d468b2f1ebd9b76882398c30bedae6c0c8bc6fdb4c506b52f`;
this identifies the artifact, not a public download location.

Public retrieval instructions and exact historical identities remain incomplete
for some checkpoint-backed families. These are explicit reproduction limitations,
not installation failures to work around with arbitrary weights. Start with a
checkpoint-free example if the required resource is unavailable.
