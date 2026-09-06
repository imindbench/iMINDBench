"""Check recipe composition, grid selection and execution behavior."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from imindbench import launch
from imindbench.utils.pipeline_contracts import validate_eval_config

CONF = Path(launch.__file__).parent / "conf"


def _args(recipe, dataset, *extra):
    return launch.parser().parse_args(
        [
            "--recipe",
            recipe,
            "--dataset",
            dataset,
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
def test_recipe_configs_compose_and_pass_runtime_contract(family, dataset):
    jobs = launch.build_commands(_args(family, dataset, "--task", "onset"))
    # Exercise every distinct model/preprocessor/sweep combination; target/fold
    # population membership is checked separately without loading recordings.
    seen = set()
    with initialize_config_dir(config_dir=str(CONF), version_base="1.1"):
        for job in jobs:
            overrides = job["command"][3:]
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


def test_population_filter_matches_manifest_and_dry_run_is_read_only(tmp_path):
    args = _args("hold_in", "neuroprobev2")
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


def test_external_config_cli_composes_without_running_experiments(tmp_path):
    config = tmp_path / "config"
    (config / "paths").mkdir(parents=True)
    (config / "paths/local.yaml").write_text((CONF / "paths/example.yaml").read_text())
    args = _args(
        "baselines",
        "neuroprobev2",
        "--model",
        "logistic",
        "--limit",
        "1",
        "--config-dir",
        str(config),
    )
    args.paths = "local"
    args.output_root = tmp_path / "runs"
    job = launch.build_commands(args)[0]
    result = subprocess.run(
        [*job["command"], "--cfg", "job"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "dataset_root: /path/to/brainsets/processed" in result.stdout
    assert not args.output_root.exists()


@pytest.mark.parametrize(
    "override",
    [
        "model=cnn",
        "dataset.task=speech",
        "hydra.run.dir=/elsewhere",
        "runtime.overwrite=true",
    ],
)
def test_identity_overrides_rejected(override):
    with pytest.raises(ValueError, match="selection flags"):
        launch.build_commands(_args("baselines", "neuroprobev2", "--set", override))


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


@pytest.mark.parametrize(
    "key", ["dataset.provider", "dataset.subset_tier", "model.name", "model.device"]
)
def test_model_overrides_cannot_change_grid_identity(tmp_path, key):
    from omegaconf import OmegaConf

    recipe = OmegaConf.load(launch.RECIPE_DIR / "paper_multistft.yaml")
    recipe.datasets.neuroprobev2.models.logistic.overrides[key] = "changed"
    path = tmp_path / "recipe.yaml"
    OmegaConf.save(recipe, path)
    with pytest.raises(ValueError, match="Invalid model-specific override"):
        launch.load_recipe(str(path), "neuroprobev2")


def test_model_settings_are_scoped_and_caller_tuning_takes_precedence(tmp_path):
    from omegaconf import OmegaConf

    recipe = OmegaConf.load(launch.RECIPE_DIR / "baselines.yaml")
    recipe.datasets.neuroprobev2.models.mlp.overrides = {
        "model.learning_rate": 0.02,
        "dataset.max_train_samples_per_subject": 12,
    }
    path = tmp_path / "recipe.yaml"
    OmegaConf.save(recipe, path)
    args = _args(str(path), "neuroprobev2", "--model", "mlp", "--limit", "1")
    first = _overrides(launch.build_commands(args)[0]["command"])
    assert first["model.learning_rate"] == "0.02"
    assert first["dataset.max_train_samples_per_subject"] == "12"
    args.overrides = ["model.learning_rate=0.03"]
    tuned = _overrides(launch.build_commands(args)[0]["command"])
    assert tuned["model.learning_rate"] == "0.03"
    args.model = ["logistic"]
    args.overrides = []
    other = _overrides(launch.build_commands(args)[0]["command"])
    assert "model.learning_rate" not in other
    assert other["dataset.max_train_samples_per_subject"] == "auto"
