from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (ROOT / "config" / "brainsets_smoke_manifest.json").read_text(encoding="utf-8")
)


def _load_pipeline_module(pipeline_id):
    from torch_brain.pipeline._cli.utils import PIPELINES_PATH

    path = PIPELINES_PATH / pipeline_id / "pipeline.py"
    spec = importlib.util.spec_from_file_location(
        f"imindbench_public_contract_{pipeline_id}", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


@pytest.mark.parametrize("pipeline_id", MANIFEST["pipelines"])
def test_prepare_uses_uv_isolation_and_forwards_options(tmp_path, pipeline_id):
    from torch_brain.pipeline._cli.cli import cli

    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    with patch("torch_brain.pipeline._cli.cli_prepare.subprocess.run") as run:
        run.return_value = MagicMock(returncode=0)
        result = CliRunner().invoke(
            cli,
            [
                "prepare",
                pipeline_id,
                "--raw-dir",
                str(raw),
                "--processed-dir",
                str(processed),
                "--list",
            ],
        )
    assert result.exit_code == 0, result.output
    command = run.call_args.args[0]
    assert command[:2] == ["uv", "run"]
    assert "--isolated" in command and "--no-project" in command
    assert "--list" in command
    assert any(part.endswith("pipeline.py") for part in command)


@pytest.mark.parametrize("pipeline_id", ["keles_byd_2024", "berezutskaya_pippi_2022"])
def test_movie_pipeline_labels_dir_is_optional(pipeline_id):
    module = _load_pipeline_module(pipeline_id)

    args = module.parser.parse_args([])

    assert args.labels_dir is None
    assert module._resolve_labels_dir(args.labels_dir) == module.PIPELINE_DIR / "labels"


@pytest.mark.parametrize(
    ("pipeline_id", "input_name"),
    [
        (
            "keles_byd_2024",
            "sub-CS48_ses-P48CSR1_behavior+ecephys.nwb",
        ),
        (
            "berezutskaya_pippi_2022",
            "sub-1_ses-iemu_task-film_acq-ecog_run-1_ieeg.vhdr",
        ),
    ],
)
def test_movie_pipeline_process_routes_default_through_labels_resolver(
    tmp_path, pipeline_id, input_name
):
    module = _load_pipeline_module(pipeline_id)
    args = module.parser.parse_args([])
    pipeline = module.Pipeline(tmp_path / "raw", tmp_path / "processed", args)
    packaged_labels = module.PIPELINE_DIR / "labels"

    with (
        patch.object(
            module, "_resolve_labels_dir", return_value=packaged_labels
        ) as resolver,
        patch.object(module, "process_file") as process_file,
    ):
        pipeline.process(tmp_path / input_name)

    resolver.assert_called_once_with(None)
    assert process_file.call_count == 1
    labels_dir = process_file.call_args.kwargs["labels_dir"]
    assert Path(labels_dir) == packaged_labels
