"""Check native experiment configs, shell selections and shared grid execution."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from imindbench import launch
from imindbench.utils.pipeline_contracts import validate_eval_config

CONF = launch.CONF_DIR
ROOT = Path(__file__).resolve().parents[1]


def _args(*extra):
    return launch.parser().parse_args(
        [
            "--dataset",
            "neuroprobev2",
            "--model",
            "logistic",
            "--preprocessor",
            "laplacian_multi_stft_2048Hz",
            "--experiment",
            "baseline",
            "--output-root",
            "/runs",
            "--paths",
            "example",
            "--device",
            "cpu",
            *extra,
        ]
    )


def _overrides(command):
    return {
        token.split("=", 1)[0].lstrip("+"): token.split("=", 1)[1]
        for token in command
        if "=" in token
    }


@pytest.mark.parametrize(
    "family,dataset",
    [
        (family, dataset)
        for family in [
            "baselines",
            "brainbert",
            "barista",
            "stft_sweep",
            "hold_in",
            "multisource",
            "sample_efficiency",
            "paper_multistft",
            "paper_brainbert_stft",
            "paper_htnet500",
            "paper_diver",
        ]
        for dataset in ["neuroprobev2", "kelesbyd2024", "berezutskayapippi2022"]
        if family != "sample_efficiency" or dataset == "neuroprobev2"
    ],
)
def test_family_shell_configs_compose_and_pass_runtime_contract(
    tmp_path, family, dataset
):
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/run_experiments.sh"),
            family,
            dataset,
            "--paths",
            "example",
            "--device",
            "cpu",
            "--task",
            "onset",
            "--output-root",
            str(tmp_path / "runs"),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    seen = set()
    with initialize_config_dir(config_dir=str(CONF), version_base="1.1"):
        for line in result.stdout.splitlines():
            overrides = shlex.split(line)[3:]
            signature = tuple(
                v
                for v in overrides
                if not any(
                    v.startswith(k)
                    for k in [
                        "dataset.test_subject=",
                        "dataset.test_session=",
                        "hydra.run.dir=",
                    ]
                )
            )
            if signature in seen:
                continue
            seen.add(signature)
            cfg = compose(config_name="config", overrides=overrides)
            validate_eval_config(cfg)
            if cfg.model.name == "htnet":
                assert "wav" in _overrides(overrides)["preprocessor"]
    assert seen
    assert not (tmp_path / "runs").exists()


def test_population_filter_matches_manifest_and_dry_run_is_read_only(tmp_path):
    args = _args(
        "--model",
        "popt",
        "--experiment",
        "decodable",
        "--regime",
        "hold-in-session",
        "--unit-set",
        "scaling",
        "--decodable-rule",
        "stft_or_htnet_500hz_val_mean0p60",
    )
    args.output_root = tmp_path / "runs"
    jobs = launch.build_commands(args)
    population = json.loads(
        (
            launch.POPULATION_DIR / "stft_or_htnet_500hz_val_mean0p60/neuroprobev2.json"
        ).read_text()
    )
    expected = {
        (task, target)
        for task, entry in population["tasks"].items()
        for target in entry["subject_sessions"]
    }
    actual = {
        (_overrides(job["command"])["dataset.task"], Path(job["run_dir"]).name)
        for job in jobs
    }
    assert actual == expected
    assert not args.output_root.exists()


def test_empty_decodable_tasks_are_skipped(tmp_path):
    (tmp_path / "neuroprobev2.json").write_text(
        json.dumps(
            {
                "tasks": {
                    "onset": {"subject_sessions": []},
                    "speech": {"subject_sessions": ["sub1_sess1"]},
                }
            }
        )
    )
    args = _args(
        "--decodable-rule",
        "stft_or_htnet_500hz_val_mean0p60",
        "--decodable-dir",
        str(tmp_path),
        "--task",
        "onset",
        "speech",
    )
    jobs = launch.build_commands(args)
    assert len(jobs) == 1
    assert _overrides(jobs[0]["command"])["dataset.task"] == "speech"
    args.task = ["onset"]
    with pytest.raises(ValueError, match="No evaluations remain"):
        launch.build_commands(args)


def test_external_config_cli_composes_without_running_experiments(tmp_path):
    config = tmp_path / "config"
    (config / "paths").mkdir(parents=True)
    (config / "experiment").mkdir()
    (config / "paths/local.yaml").write_text((CONF / "paths/example.yaml").read_text())
    (config / "experiment/custom.yaml").write_text(
        "# @package _global_\nmodel:\n  max_iter: 17\n"
    )
    args = _args(
        "--limit",
        "1",
        "--config-dir",
        str(config),
        "--paths",
        "local",
        "--experiment",
        "custom",
    )
    args.output_root = tmp_path / "runs"
    job = launch.build_commands(args)[0]
    result = subprocess.run(
        [*job["command"], "--cfg", "job"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert "dataset_root: /path/to/brainsets/processed" in result.stdout
    assert "max_iter: 17" in result.stdout
    assert not args.output_root.exists()


@pytest.mark.parametrize(
    "override",
    [
        "model=cnn",
        "dataset.task=speech",
        "hydra.run.dir=/elsewhere",
        "runtime.overwrite=true",
        "model.name=other",
        "dataset.provider=other",
        "dataset.subset_tier=full",
        "model.device=cuda:1",
    ],
)
def test_identity_overrides_rejected(override):
    with pytest.raises(ValueError, match="selection flags"):
        launch.build_commands(_args("--set", override))


def test_cli_tuning_overrides_native_experiment_settings():
    args = _args(
        "--model",
        "mlp",
        "--experiment",
        "paper_multistft/mlp",
        "--limit",
        "1",
        "--set",
        "model.tol=0.02",
        "--set",
        "model.tol=0.03",
    )
    job = launch.build_commands(args)[0]
    with initialize_config_dir(config_dir=str(CONF), version_base="1.1"):
        cfg = compose(config_name="config", overrides=job["command"][3:])
    assert cfg.model.tol == 0.03
    assert cfg.runner.num_workers == 4


@pytest.mark.parametrize(
    "options",
    [
        ["--sweep", "model.tol=0.1,0.1"],
        ["--sweep", "model.tol=0.1,0.2", "--set", "model.tol=0.3"],
        ["--limit", "0"],
    ],
)
def test_ambiguous_or_invalid_grids_are_rejected(options):
    with pytest.raises(ValueError):
        launch.build_commands(_args(*options))


def test_execution_resume_and_existing_run_preflight(tmp_path):
    result = tmp_path / "job" / "population.json"
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(result)!r}).write_text('{{}}')",
    ]
    job = {"command": command, "run_dir": str(result.parent), "result": str(result)}
    assert launch.execute_commands([job], tmp_path) == 0
    assert result.is_file()
    before = result.stat().st_mtime_ns
    assert launch.execute_commands([job], tmp_path, resume=True) == 0
    assert result.stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="Existing run"):
        launch.execute_commands([job], tmp_path)
    different = {
        **job,
        "command": [sys.executable, "-c", "raise AssertionError('must not run')"],
    }
    with pytest.raises(ValueError, match="same launch.json"):
        launch.execute_commands([different], tmp_path, resume=True)
    result.write_text("partial result")
    with pytest.raises(ValueError, match="Unverified or changed result"):
        launch.execute_commands([job], tmp_path, resume=True)
