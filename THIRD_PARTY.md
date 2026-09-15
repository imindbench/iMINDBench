# Third-party code

Original iMINDBench contributions are offered under Apache-2.0 (see
`LICENSE.txt`). Incorporated third-party code retains the terms documented below.

Inspected revisions and reference versions identify the sources checked, which
may differ from the revisions originally copied.

| Component | License status |
| --- | --- |
| BaRISTA | USC educational, research, and non-profit terms; see below. |
| DIVER-1-specific code | No license grant found in the inspected source; permission unresolved. |
| PopT-specific code | MIT; upstream copyright and license preserved in `LICENSES/PopT-LICENSE.txt`. |
| uni2ts portions within DIVER | Apache-2.0; Salesforce notices preserved below and in the encoder. |
| Tutorial/PyTorch portions within PopT | Tutorial displays CC BY-NC 4.0; PyTorch portions retain their BSD-style terms. |
| SciPy normalization portions through PopT | BSD-3-Clause; upstream notice preserved in `LICENSES/SciPy-LICENSE.txt`. |

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
  model/utility sources contain no license grant. Permission for DIVER-specific
  code remains unresolved; the uni2ts terms below cover only uni2ts-derived code.

The local implementation adapts the upstream model components to iMINDBench's
batch interface, classification heads, checkpoint loading, and MuP shape cache.
The YAML records the benchmark training recipe.

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

---

## PopulationTransformer (PopT)

- **Upstream:** https://github.com/czlwang/PopulationTransformer
- **Upstream revision used for the port:** unconfirmed
- **Inspected revision:** [`d237755bbdea123c60bface0cecbb48bab6f42de`](https://github.com/czlwang/PopulationTransformer/tree/d237755bbdea123c60bface0cecbb48bab6f42de)
- **Paper:** [Population Transformer: Learning Population-level Representations of Neural Activity](https://arxiv.org/abs/2406.03044)
- **Upstream license:** MIT for original PopulationTransformer code and
  documentation; incorporated third-party material retains its own terms.
- **Copyright:** Copyright (c) 2024-2026 PopulationTransformer contributors.
- **License copy:** [`LICENSES/PopT-LICENSE.txt`](LICENSES/PopT-LICENSE.txt), copied
  verbatim from the inspected revision's root `LICENSE`.
- **Upstream scope and notices:** [`THIRD_PARTY.md`](https://github.com/czlwang/PopulationTransformer/blob/d237755bbdea123c60bface0cecbb48bab6f42de/THIRD_PARTY.md).

The following paths are relative to `imindbench/` locally and to the inspected
PopulationTransformer repository upstream:

| Local integration or adapted component | Upstream source reference |
| --- | --- |
| `models/popt_model.py` | `models/pt_downstream_model.py`, `models/pt_model_custom.py` |
| `models/popt_components/transformer_input.py`, `models/popt_components/positional_encoding.py` | `models/transformer_encoder_input.py` |
| `models/popt_components/transformer_layers.py` | Transformer classes in `models/pt_model_custom.py`; earlier sources below |
| `schedulers/ramp_up.py` | `schedulers/ramp_up.py` |
| `preprocessors/stft_preprocessor.py` (`zscore` normalization) | `preprocessors/stft.py`; SciPy-derived portions below |

Local adaptations include the benchmark batch interface, multiclass heads,
checkpoint conversion, variable-channel handling, separated component modules,
and configurable scheduler intervals. `conf/model/popt.yaml` records the benchmark
recipe.

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
  in [`LICENSES/PyTorch-LICENSE.txt`](LICENSES/PyTorch-LICENSE.txt).
- **SciPy normalization:** the NumPy path in
  `preprocessors/stft_preprocessor.py::zscore` retains the normalization pattern
  from PopT's `preprocessors/stft.py`, which credits
  [`scipy/stats/_stats_py.py` at v1.9.0](https://github.com/scipy/scipy/blob/v1.9.0/scipy/stats/_stats_py.py).
  Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
  The BSD-3-Clause notice is copied verbatim from PopT's inspected revision in
  [`LICENSES/SciPy-LICENSE.txt`](LICENSES/SciPy-LICENSE.txt). Local changes include
  a Torch tensor path, array conversion, and the `dim` argument name; the helper
  replaces zero standard deviations with one.
- **Positional encoding:** the official PopT implementation additionally credits
  [a PyTorch forum discussion](https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308/2).
  The cited post contains no code, so it does not identify the precise snippet
  used. The original snippet, revision, and applicable terms for any forum-specific
  contribution remain unverified.
- **Warmup dependency:** the scheduler uses `GradualWarmupScheduler` from
  [pytorch-gradual-warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr),
  installed separately at the revision pinned in `pyproject.toml`. Its package
  terms remain separate from the MIT license for PopT's scheduler adapter.
