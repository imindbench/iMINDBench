# Third-party code

This repository is licensed under Apache-2.0 (see `LICENSE.txt`), except for the
third-party components listed here. Each component remains under its upstream
license; the repository's Apache license does not replace those terms.

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
