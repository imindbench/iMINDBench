"""Exercise the release interfaces outside a Git checkout."""

import importlib.util
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_launcher_composes_external_config_without_git(tmp_path):
    scripts = tmp_path / "installation" / "scripts"
    shutil.copytree(ROOT / "imindbench/scripts", scripts)
    config = tmp_path / "config"
    (config / "paths").mkdir(parents=True)
    shutil.copy(
        ROOT / "imindbench/conf/paths/example.yaml", config / "paths/local.yaml"
    )
    output = tmp_path / "runs"
    env = {
        **os.environ,
        "IMINDBENCH_CONFIG_DIR": str(config),
        "IMINDBENCH_OUTPUT_ROOT": str(output),
        "PATHS_CFG": "local",
        "BARISTA_CHECKPOINT": str(tmp_path / "unused.ckpt"),
        "TASKS": "onset",
        "SUBJECT_TRIALS_STR": "1 1",
        "TMPDIR": str(tmp_path / "temp"),
        "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]),
    }
    result = subprocess.run(
        ["bash", str(scripts / "run_neuroprobev2_barista.sh"), "--cfg", "job"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "dataset_root: /path/to/brainsets/processed" in result.stdout
    assert output.is_dir()
    assert not list((tmp_path / "installation").rglob("outputs"))
    assert not list(output.rglob("population_*.json"))


def test_submission_is_reproducible_and_excludes_local_files(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "build_submission", ROOT / "scripts/build_submission.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    (source / "config").mkdir(parents=True)
    (source / "config/submission_files.json").write_text(json.dumps(["README.md"]))
    (source / "README.md").write_text("Reviewer instructions\n")
    (source / "private.txt").write_text("Must not be exported")
    first, second = tmp_path / "first.tar.gz", tmp_path / "second.tar.gz"
    module.build_submission(source, first)
    module.build_submission(source, second)
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first) as archive:
        assert archive.getnames() == ["imindbench/README.md", "imindbench/SHA256SUMS"]
        assert all(member.uid == 0 and member.mtime == 0 for member in archive)
    with pytest.raises(FileExistsError):
        module.build_submission(source, first)
    (source / "README.md").write_text("/" + "home" + "/author/private")
    with pytest.raises(ValueError, match="Personal path"):
        module.build_submission(source, tmp_path / "private.tar.gz")
    (source / "README.md").unlink()
    (source / "README.md").symlink_to(source / "private.txt")
    with pytest.raises(ValueError, match="symlinks"):
        module.build_submission(source, tmp_path / "symlink.tar.gz")
