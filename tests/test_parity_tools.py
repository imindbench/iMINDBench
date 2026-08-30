from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "artifacts" / "parity_reference" / "manifest.json"
SPEC = importlib.util.spec_from_file_location(
    "parity_tools", ROOT / "scripts" / "parity_tools.py"
)
assert SPEC is not None and SPEC.loader is not None
parity_tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parity_tools)


def _multi_stft_preprocess(sampling_rate: int) -> dict[str, object]:
    scale = sampling_rate / 2048
    return {
        "name": "laplacian_multi_stft",
        "chain": [
            {
                "name": "time_domain_filter",
                "sampling_rate": sampling_rate,
                "high_gamma": False,
                "notch_q": 30,
            },
            {"name": "laplacian_rereference", "remove_non_laplacian": True},
            {
                "name": "multi_stft",
                "sampling_rate": sampling_rate,
                "hop_length": round(128 * scale),
                "window": "hann",
                "normalizing": "none",
                "windows": [
                    {
                        "label": "low",
                        "nperseg": round(1024 * scale),
                        "min_frequency": 2,
                        "max_frequency": 40,
                    },
                    {
                        "label": "mid",
                        "nperseg": round(512 * scale),
                        "min_frequency": 20,
                        "max_frequency": 150,
                    },
                    {
                        "label": "high",
                        "nperseg": round(256 * scale),
                        "min_frequency": 80,
                        "max_frequency": 250,
                    },
                ],
            },
            {"name": "standardize", "mode": "per_channel_samples_time_pooled"},
        ],
    }


def _candidate(manifest: dict[str, object], case_id: str) -> dict[str, object]:
    record = parity_tools.load_reference_record(manifest, case_id)
    identity = record["identity"]
    return {
        "model_name": identity["model_name"],
        "evaluation_results": {
            identity["recording_key"]: {
                identity["scope"]: {
                    identity["time_bin_key"]: {
                        "time_bin_start": identity["time_bin_start"],
                        "time_bin_end": identity["time_bin_end"],
                        "folds": record["folds"],
                    }
                }
            }
        },
        "config": {
            "preprocess": _multi_stft_preprocess(2048),
            "seed": identity["seed"],
            "subject_id": identity["subject"],
            "trial_id": identity["session"],
            "eval_name": identity["task"],
            "splits_type": identity["regime"],
            "model_name": identity["model_name"],
        },
    }


@pytest.fixture
def manifest() -> dict[str, object]:
    loaded = parity_tools.load_manifest(MANIFEST_PATH)
    assert loaded["source_code"]["commit"] == "726d96978c3e573c7a31e2e63b18c905e4c15d6b"
    return loaded


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_build_command_is_dry_run_and_requires_fresh_output(tmp_path, manifest):
    data_root = tmp_path / "data"
    data_root.mkdir()
    output_root = tmp_path / "new-output"
    command = parity_tools.build_case_command(
        manifest,
        "neuroprobev2_logistic_multistft_onset_sub1_sess1",
        data_root=data_root,
        output_root=output_root,
        checkpoint_map={},
        resource_map={},
    )
    assert command[1:3] == ["-m", "imindbench.run_eval"]
    assert "dataset.task=onset" in command
    assert f"hydra.run.dir={output_root}" in command
    assert not output_root.exists()

    output_root.mkdir()
    with pytest.raises(ValueError, match="must be fresh"):
        parity_tools.build_case_command(
            manifest,
            "neuroprobev2_logistic_multistft_onset_sub1_sess1",
            data_root=data_root,
            output_root=output_root,
            checkpoint_map={},
            resource_map={},
        )


def test_checkpoint_case_requires_caller_mapping(tmp_path, manifest):
    data_root = tmp_path / "data"
    data_root.mkdir()
    with pytest.raises(ValueError, match="popt_checkpoint"):
        parity_tools.build_case_command(
            manifest,
            "pippi_popt_multistft_speech_sub1_sess1",
            data_root=data_root,
            output_root=tmp_path / "output",
            checkpoint_map={},
            resource_map={},
        )


def test_compare_reports_pass_fail_and_missing(tmp_path, manifest):
    case_id = "neuroprobev2_logistic_multistft_onset_sub1_sess1"
    candidate = _candidate(manifest, case_id)
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, candidate)
    pass_report = parity_tools.compare_case(
        manifest, case_id, candidate_path, checkpoint_map={}
    )
    assert pass_report["status"] == "PASS"
    assert (
        pass_report["candidate_sha256"]
        == hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    )
    assert len(pass_report["source_result_sha256"]) == 64
    assert pass_report["reference_record_verified"] is True
    assert pass_report["parity_scope"] == "metric-config-record"
    assert pass_report["execution_provenance_verified"] is False

    candidate["evaluation_results"]["btbank1_1"]["population"][
        "one_second_after_onset"
    ]["folds"][0]["test_roc_auc"] = 0.0
    _write_json(candidate_path, candidate)
    assert (
        parity_tools.compare_case(manifest, case_id, candidate_path, checkpoint_map={})[
            "status"
        ]
        == "FAIL"
    )
    assert (
        parity_tools.compare_case(
            manifest, case_id, tmp_path / "absent.json", checkpoint_map={}
        )["status"]
        == "MISSING"
    )


def test_missing_reference_checkpoint_is_not_comparable(tmp_path, manifest):
    case_id = "pippi_popt_multistft_speech_sub1_sess1"
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(_candidate(manifest, case_id)), encoding="utf-8"
    )
    checkpoint = tmp_path / "popt.ckpt"
    checkpoint.write_bytes(b"caller supplied checkpoint")
    report = parity_tools.compare_case(
        manifest,
        case_id,
        candidate_path,
        checkpoint_map={"popt_checkpoint": checkpoint},
    )
    assert report["status"] == "NOT-COMPARABLE"
    assert report["checkpoint"]["reference_sha256"] is None
    assert len(report["checkpoint"]["supplied_sha256"]) == 64


def test_mismatch_on_checkpoint_blocked_case_is_fail(tmp_path, manifest):
    case_id = "pippi_popt_multistft_speech_sub1_sess1"
    candidate = _candidate(manifest, case_id)
    candidate["evaluation_results"]["btbank1_1"]["population"][
        "one_second_after_onset"
    ]["folds"][0]["test_accuracy"] = 0.0
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, candidate)
    checkpoint = tmp_path / "popt.ckpt"
    checkpoint.write_bytes(b"caller supplied checkpoint")
    report = parity_tools.compare_case(
        manifest,
        case_id,
        candidate_path,
        checkpoint_map={"popt_checkpoint": checkpoint},
    )
    assert report["status"] == "FAIL"


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), -float("inf")])
def test_candidate_rejects_non_finite_json_numbers(tmp_path, manifest, non_finite):
    case_id = "neuroprobev2_logistic_multistft_onset_sub1_sess1"
    candidate = _candidate(manifest, case_id)
    candidate["evaluation_results"]["btbank1_1"]["population"][
        "one_second_after_onset"
    ]["folds"][0]["test_accuracy"] = non_finite
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, candidate)
    with pytest.raises(ValueError, match="non-finite"):
        parity_tools.compare_case(manifest, case_id, candidate_path, checkpoint_map={})


def test_manifest_rejects_non_finite_tolerance_and_unsafe_case_id(tmp_path, manifest):
    raw = json.loads(MANIFEST_PATH.read_text())
    raw["tolerance"]["absolute"] = float("nan")
    invalid_path = tmp_path / "invalid.json"
    _write_json(invalid_path, raw)
    with pytest.raises(ValueError, match="non-finite"):
        parity_tools.load_manifest(invalid_path)

    raw = json.loads(MANIFEST_PATH.read_text())
    case = raw["cases"].pop(next(iter(raw["cases"])))
    raw["cases"]["../escape"] = case
    _write_json(invalid_path, raw)
    with pytest.raises(ValueError, match="Unsafe parity case ID"):
        parity_tools.load_manifest(invalid_path)

    raw = json.loads(MANIFEST_PATH.read_text())
    raw.pop("runner_module")
    _write_json(invalid_path, raw)
    with pytest.raises(ValueError, match="missing required keys"):
        parity_tools.load_manifest(invalid_path)

    raw = json.loads(MANIFEST_PATH.read_text())
    raw["source_code"]["commit"] = raw["source_code"]["commit"].upper()
    _write_json(invalid_path, raw)
    with pytest.raises(ValueError, match="lowercase 40-character Git SHA"):
        parity_tools.load_manifest(invalid_path)


def test_reference_rejects_tampered_hash_and_path_escape(manifest):
    case_id = "neuroprobev2_logistic_multistft_onset_sub1_sess1"
    tampered = copy.deepcopy(manifest)
    tampered["cases"][case_id]["reference_record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Reference hash mismatch"):
        parity_tools.load_reference_record(tampered, case_id)

    escaped = copy.deepcopy(manifest)
    escaped["cases"][case_id][
        "reference_record_uri"
    ] = "imindbench://artifacts/parity_reference/README.md"
    with pytest.raises(ValueError, match="escapes"):
        parity_tools.load_reference_record(escaped, case_id)


def test_reference_rejects_duplicate_or_non_integer_folds(tmp_path, manifest):
    case_id = "neuroprobev2_logistic_multistft_onset_sub1_sess1"
    original = parity_tools.load_reference_record(manifest, case_id)
    for invalid_index in (0, 1.5):
        project = tmp_path / str(invalid_index).replace(".", "_") / "project"
        record_dir = project / "artifacts" / "parity_reference" / "cases"
        record_dir.mkdir(parents=True)
        record = copy.deepcopy(original)
        record["folds"][1]["fold_idx"] = invalid_index
        record_path = record_dir / f"{case_id}.json"
        _write_json(record_path, record)
        raw_manifest = copy.deepcopy(
            {key: value for key, value in manifest.items() if key != "_manifest_path"}
        )
        raw_manifest["cases"] = {case_id: raw_manifest["cases"][case_id]}
        raw_manifest["cases"][case_id]["reference_record_sha256"] = hashlib.sha256(
            record_path.read_bytes()
        ).hexdigest()
        manifest_path = project / "artifacts" / "parity_reference" / "manifest.json"
        _write_json(manifest_path, raw_manifest)
        loaded = parity_tools.load_manifest(manifest_path)
        with pytest.raises(ValueError, match="unique non-negative integers"):
            parity_tools.load_reference_record(loaded, case_id)


def test_resource_directory_fingerprint_is_recorded(tmp_path, manifest):
    case_id = "neuroprobev2_popt_multistft_holdin_onset_sub1_sess1"
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, _candidate(manifest, case_id))
    checkpoint = tmp_path / "popt.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    resource = tmp_path / "decodable"
    resource.mkdir()
    (resource / "neuroprobev2.json").write_text("{}", encoding="utf-8")
    report = parity_tools.compare_case(
        manifest,
        case_id,
        candidate_path,
        checkpoint_map={"popt_checkpoint": checkpoint},
        resource_map={"decodable_subject_sessions_dir": resource},
    )
    assert report["status"] == "NOT-COMPARABLE"
    assert report["resources"][0]["mapping_key"] == "decodable_subject_sessions_dir"
    assert len(report["resources"][0]["supplied_sha256"]) == 64


def test_report_writer_uses_fresh_directory(tmp_path):
    reports = [{"case_id": "case", "status": "PASS", "findings": ["matched"]}]
    report_dir = tmp_path / "report"
    json_path, markdown_path = parity_tools.write_report(reports, report_dir)
    assert json.loads(json_path.read_text())["status"] == "PASS"
    assert "| case | PASS | matched |" in markdown_path.read_text()
    with pytest.raises(ValueError, match="must be fresh"):
        parity_tools.write_report(reports, report_dir)


def test_report_failure_leaves_no_partial_or_temporary_directory(tmp_path):
    report_dir = tmp_path / "report"
    reports = [{"case_id": "case", "status": "PASS", "findings": [{"not-json"}]}]
    with pytest.raises(TypeError):
        parity_tools.write_report(reports, report_dir)
    assert not report_dir.exists()
    assert list(tmp_path.glob(".report.tmp-*")) == []


def test_report_rename_failure_cleans_temporary_directory(tmp_path, monkeypatch):
    report_dir = tmp_path / "report"
    reports = [{"case_id": "case", "status": "PASS", "findings": ["matched"]}]

    def fail_rename(source, destination):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(parity_tools.os, "rename", fail_rename)
    with pytest.raises(OSError, match="simulated rename failure"):
        parity_tools.write_report(reports, report_dir)
    assert not report_dir.exists()
    assert list(tmp_path.glob(".report.tmp-*")) == []


def test_compare_cli_exit_codes(tmp_path, manifest):
    case_id = "neuroprobev2_logistic_multistft_onset_sub1_sess1"
    candidate = tmp_path / "candidate.json"
    _write_json(candidate, _candidate(manifest, case_id))
    candidate_map = tmp_path / "candidates.json"
    checkpoint_map = tmp_path / "checkpoints.json"
    resource_map = tmp_path / "resources.json"
    _write_json(candidate_map, {case_id: str(candidate)})
    _write_json(checkpoint_map, {})
    _write_json(resource_map, {})
    common = [
        "--manifest",
        str(MANIFEST_PATH),
        "compare",
        "--case",
        case_id,
        "--candidate-map",
        str(candidate_map),
        "--checkpoint-map",
        str(checkpoint_map),
        "--resource-map",
        str(resource_map),
    ]
    assert parity_tools.main([*common, "--report-dir", str(tmp_path / "pass")]) == 0
    _write_json(candidate_map, {})
    assert parity_tools.main([*common, "--report-dir", str(tmp_path / "missing")]) == 1


def test_not_comparable_cli_exit_is_nonzero(tmp_path, manifest):
    case_id = "pippi_popt_multistft_speech_sub1_sess1"
    candidate = tmp_path / "candidate.json"
    _write_json(candidate, _candidate(manifest, case_id))
    checkpoint = tmp_path / "popt.ckpt"
    checkpoint.write_bytes(b"caller checkpoint")
    mapping_values = {
        "candidates": {case_id: str(candidate)},
        "checkpoints": {"popt_checkpoint": str(checkpoint)},
        "resources": {},
    }
    mapping_paths = {}
    for name, value in mapping_values.items():
        path = tmp_path / f"{name}.json"
        _write_json(path, value)
        mapping_paths[name] = path
    exit_code = parity_tools.main(
        [
            "--manifest",
            str(MANIFEST_PATH),
            "compare",
            "--case",
            case_id,
            "--candidate-map",
            str(mapping_paths["candidates"]),
            "--checkpoint-map",
            str(mapping_paths["checkpoints"]),
            "--resource-map",
            str(mapping_paths["resources"]),
            "--report-dir",
            str(tmp_path / "report"),
        ]
    )
    assert exit_code == 1
