from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "config" / "brainsets_smoke_manifest.json").read_text(encoding="utf-8")
)


def test_public_dataset_exports_match_smoke_manifest():
    from torch_brain import datasets

    for spec in MANIFEST["datasets"].values():
        assert getattr(datasets, spec["class"]).__name__ == spec["class"]


def test_exact_public_pipeline_ids_and_assets_are_packaged():
    from torch_brain.pipeline._cli.utils import PIPELINES_PATH, get_available_brainsets

    available = set(get_available_brainsets())
    assert set(MANIFEST["pipelines"]) <= available
    for pipeline_id in ("keles_byd_2024", "berezutskaya_pippi_2022"):
        pipeline = PIPELINES_PATH / pipeline_id
        assert len(list((pipeline / "labels").glob("*.csv"))) == 30
        assert (pipeline / "brain_areas" / "brain_area_labels.csv").is_file()
