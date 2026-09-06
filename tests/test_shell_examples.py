"""Exercise the user-facing Bash examples without data or training."""

import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from imindbench.utils.pipeline_contracts import validate_eval_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script,model,device",
    [
        ("run_logistic.sh", "logistic", "cpu"),
        ("run_mlp.sh", "mlp", "cuda:0"),
        ("run_barista.sh", "barista", "cuda:0"),
    ],
)
def test_shell_example_previews_one_composable_evaluation(
    tmp_path, script, model, device
):
    # Copy outside the checkout to exercise quoting and independence from Git/cwd.
    example = tmp_path / f"copied {script}"
    shutil.copyfile(ROOT / "scripts" / script, example)
    outside = tmp_path / "outside"
    outside.mkdir()
    config = tmp_path / "local config"
    (config / "paths").mkdir(parents=True)
    shutil.copyfile(
        ROOT / "imindbench/conf/paths/example.yaml", config / "paths/local.yaml"
    )
    output = tmp_path / "run outputs"
    args = [
        "bash",
        str(example),
        "--config-dir",
        str(config),
        "--output-root",
        str(output),
        "--set",
        "model.max_iter=5",
    ]
    preview = subprocess.run(
        args, cwd=outside, text=True, capture_output=True, check=True
    )
    assert "1 evaluations; dry run" in preview.stderr
    assert len(preview.stdout.splitlines()) == 1
    command = shlex.split(preview.stdout.strip())
    # Compose the exact emitted command through the real CLI without reading data.
    composed = subprocess.run(
        [*command, "--cfg", "job", "--resolve"],
        cwd=outside,
        text=True,
        capture_output=True,
        check=True,
    )
    cfg = OmegaConf.create(composed.stdout)
    validate_eval_config(cfg)
    assert cfg.dataset.task == "onset"
    assert cfg.dataset.test_subject == 1
    assert cfg.dataset.test_session == 1
    assert cfg.model.name == model
    assert cfg.model.max_iter == 5
    assert cfg.model.device == device
    assert not output.exists()

    count = subprocess.run(
        [*args, "--count"], cwd=outside, text=True, capture_output=True, check=True
    )
    assert count.stdout.strip() == "1"
    invalid = subprocess.run(
        [*args, "--resume"], cwd=outside, text=True, capture_output=True
    )
    assert invalid.returncode == 2
    assert "--resume requires --execute" in invalid.stderr


def test_family_script_filters_models_and_replaces_task_selection(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/run_experiments.sh"),
            "barista",
            "neuroprobev2",
            "barista",
            "--paths",
            "example",
            "--task",
            "speech",
            "--target",
            "sub1_sess1",
            "--output-root",
            str(tmp_path / "runs"),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert len(result.stdout.splitlines()) == 1
    assert "dataset.task=speech" in shlex.split(result.stdout.strip())
    invalid = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/run_experiments.sh"),
            "paper_diver",
            "neuroprobev2",
            "mlp",
            "--output-root",
            str(tmp_path / "runs"),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert invalid.returncode == 2
    assert "Model must be one of: diver" in invalid.stderr


@pytest.mark.parametrize(
    "dataset,pair_count",
    [
        ("neuroprobev2", 5),
        ("kelesbyd2024", 29),
        ("berezutskayapippi2022", 5),
    ],
)
@pytest.mark.parametrize("family", ["barista", "brainbert"])
def test_pretrained_families_cover_all_tasks_and_targets(
    tmp_path, dataset, pair_count, family
):
    output = tmp_path / "runs"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/run_experiments.sh"),
            family,
            dataset,
            "--paths",
            "example",
            "--output-root",
            str(output),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    units = set()
    for line in result.stdout.splitlines():
        fields = dict(
            token.split("=", 1) for token in shlex.split(line) if "=" in token
        )
        units.add(
            (
                fields["dataset.task"],
                fields["dataset.test_subject"],
                fields["dataset.test_session"],
            )
        )
    catalog = OmegaConf.load(ROOT / "imindbench/conf/units/catalog.yaml")
    expected = {
        (task, str(subject), str(session))
        for task in catalog.tasks
        for subject, session in catalog.datasets[dataset].targets.all
    }
    assert units == expected
    assert len(result.stdout.splitlines()) == len(units) == 15 * pair_count
    assert not output.exists()
