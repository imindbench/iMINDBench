# Third-party code

Original iMINDBench contributions are offered under Apache-2.0 (see
`LICENSE.txt`). Incorporated third-party code retains its applicable terms;
this repository does not grant rights to upstream code whose license is unknown.

The DIVER and PopT source review below was performed on 2026-09-13. It records
verified notices and unresolved permissions, not legal clearance for a release.
An "inspected revision" identifies the source checked during this review; it
does not establish which revision was originally copied.

| Component | License status |
| --- | --- |
| BaRISTA | USC educational, research, and non-profit terms; see below. |
| DIVER-1-specific code | No license grant found in the inspected source; permission unresolved. |
| PopT-specific code | No license grant found in the inspected source; permission unresolved. |
| uni2ts portions within DIVER | Apache-2.0; Salesforce notices preserved below and in the encoder. |
| Tutorial/PyTorch portions within PopT | Tutorial displays CC BY-NC 4.0; PyTorch portions retain their BSD-style terms. |

---

## BaRISTA

- **Files:** `imindbench/models/barista_model.py`,
  `imindbench/models/barista_components/*.py`
- **Upstream:** https://github.com/ShanechiLab/BaRISTA
- **Upstream commit:** not recorded at port time; license verified at upstream
  commit `83b27375eba60e9eba9da4e7dd8fb283baace376` (2026-08-18)
- **Upstream license:** USC educational, research, and non-profit license
- **License copy:** `LICENSES/BaRISTA-LICENSE.md`

BaRISTA is not covered by this repository's Apache-2.0 license. Its source may be
used, copied, modified, and distributed for educational, research, and non-profit
purposes when the required USC notice accompanies every copy. Commercial use
requires separate permission from the USC Stevens Center for Innovation.

`models/barista_components/TSEncoder2D.py` also cites the `pytorch/vision`
DenseNet weight-initialisation idiom in a docstring. That is a reference to a
convention rather than a copy, so it raises no separate licensing question.
`atlas.py` cites the Destrieux/FreeSurfer papers for parcel names, likewise.

---

## DIVER-1

- **Files:** `imindbench/models/diver_model.py`,
  `imindbench/models/diver_components/*.py`; benchmark configuration in
  `imindbench/conf/model/diver.yaml`
- **Upstream:** https://github.com/DIVER-Project/DIVER-1
- **Upstream revision used for the port:** unconfirmed
- **Inspected revision:** [`fae4d7c5a58f2f795ce767939ad191d9c7ba49b8`](https://github.com/DIVER-Project/DIVER-1/tree/fae4d7c5a58f2f795ce767939ad191d9c7ba49b8)
- **Source references:** [model components](https://github.com/DIVER-Project/DIVER-1/tree/fae4d7c5a58f2f795ce767939ad191d9c7ba49b8/models)
  and [utilities](https://github.com/DIVER-Project/DIVER-1/tree/fae4d7c5a58f2f795ce767939ad191d9c7ba49b8/utils).
  Local model component filenames follow upstream; `checkpoint_utils.py` adapts
  `utils/checkpoint.py`, and `mup_utils.py` adapts `utils/mup_utils.py`.
- **Paper:** [DIVER-1: Scaling Intracranial EEG Foundation Models for Transferable Representations](https://arxiv.org/abs/2512.19097)
- **Upstream license:** no grant found. The complete tracked tree at the inspected
  revision contains no license/copying/notice file, and the README and inspected
  model/utility sources contain no license grant. No DIVER license copy is
  supplied because none was found; the uni2ts terms below cover only those portions.

The local implementation adapts the upstream model components to iMINDBench's
batch interface, classification heads, checkpoint loading, and MuP shape cache.
The YAML records the benchmark training recipe. These adaptations do not establish
permission for the DIVER-specific contributions; an applicable license or written
permission from the rights holders still needs to be recorded.

### uni2ts code incorporated through DIVER

`imindbench/models/diver_components/original_moirai_encoder.py` contains adapted
classes from Salesforce's [uni2ts](https://github.com/SalesforceAIResearch/uni2ts),
including attention, feed-forward, normalization, positional-projection, and
transformer classes. The local file combines modules and includes DIVER/iMINDBench
changes to imports, typing, and attention behavior.

- **Inspected revision:** `cfd46d4510ed8896f263116f32928eede05b0a75`
- **Source:** [`src/uni2ts/module/`](https://github.com/SalesforceAIResearch/uni2ts/tree/cfd46d4510ed8896f263116f32928eede05b0a75/src/uni2ts/module),
  specifically `attention.py`, `ffn.py`, `norm.py`, `position/attn_bias.py`,
  `position/attn_projection.py`, and `transformer.py`.
- **Notices:** Copyright (c) 2024, Salesforce, Inc. (module headers);
  Copyright (c) 2023 Salesforce, Inc. (repository license).
- **License:** Apache-2.0; full upstream text in
  [`LICENSES/uni2ts-LICENSE.txt`](LICENSES/uni2ts-LICENSE.txt).

The Salesforce notice applies to the uni2ts-derived portions. It does not license
the otherwise unresolved DIVER-specific additions.

---

## PopulationTransformer (PopT)

- **Upstream:** https://github.com/czlwang/PopulationTransformer
- **Upstream revision used for the port:** unconfirmed
- **Inspected revision:** [`dadb55b21daf2809b123d49a9594ff6eebea2c40`](https://github.com/czlwang/PopulationTransformer/tree/dadb55b21daf2809b123d49a9594ff6eebea2c40)
- **Paper:** [Population Transformer: Learning Population-level Representations of Neural Activity](https://arxiv.org/abs/2406.03044)
- **Upstream license:** no grant found. The complete tracked tree at the inspected
  revision contains no license/copying/notice file, and the README and inspected
  model/scheduler sources contain no license grant. No PopT license copy is supplied
  because none was found.

The following paths are relative to `imindbench/` locally and to the inspected
PopulationTransformer repository upstream:

| Local integration or adapted component | Upstream source reference |
| --- | --- |
| `models/popt_model.py` | `models/pt_downstream_model.py`, `models/pt_model_custom.py` |
| `models/popt_components/transformer_input.py`, `models/popt_components/positional_encoding.py` | `models/transformer_encoder_input.py` |
| `models/popt_components/transformer_layers.py` | Transformer classes in `models/pt_model_custom.py`; earlier sources below |
| `schedulers/ramp_up.py` | `schedulers/ramp_up.py` |

Local adaptations include the benchmark batch interface, multiclass heads,
checkpoint conversion, variable-channel handling, separated component modules,
and configurable scheduler intervals. `conf/model/popt.yaml` records the benchmark
recipe; it is not evidence of exact upstream reproduction. The local linear
baseline and `brainbert_encoder.py` integration helper are not asserted to be
files copied from this official PopT repository.

### Earlier sources within PopT

- **Transformer tutorial:** `models/popt_components/transformer_layers.py` adapts
  the attention-returning encoder shown in Buomsoo Kim's
  ["Attention in Neural Networks - 21. Transformer (5)"](https://buomsoo-kim.github.io/attention/2020/04/27/Attention-mechanism-21.md/)
  (2020-04-27), also credited by PopT upstream. The site displays
  "© 2022 Buomsoo Kim" and [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
  ([legal terms](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en)).
  Preserve attribution, indicate modifications, and observe the noncommercial
  restriction for material covered by that grant. Local changes include module
  separation, formatting, and documentation. No endorsement is implied.
- **PyTorch portions:** the tutorial adapts PyTorch's transformer implementation.
  The local encoder retains the corresponding layer structure and helper methods.
  Reference source: [`torch/nn/modules/transformer.py` at v1.5.0](https://github.com/pytorch/pytorch/blob/v1.5.0/torch/nn/modules/transformer.py).
  The full copyright notices, BSD-style conditions, and disclaimer are preserved
  in [`LICENSES/PyTorch-LICENSE.txt`](LICENSES/PyTorch-LICENSE.txt). This reference
  version does not establish the precise historical copy revision. Its license
  does not establish permission for later authors' additions.
- **Positional encoding:** the official PopT implementation additionally credits
  [a PyTorch forum discussion](https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308/2).
  The cited post contains no code, so it does not identify the precise snippet
  used. The original snippet, revision, and applicable terms for any forum-specific
  contribution remain unverified; this is part of the unresolved PopT provenance.
- **Warmup dependency:** the scheduler uses `GradualWarmupScheduler` from
  [pytorch-gradual-warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr),
  installed separately at the revision pinned in `pyproject.toml`. Its package
  terms do not establish permission for PopT's scheduler adapter.
