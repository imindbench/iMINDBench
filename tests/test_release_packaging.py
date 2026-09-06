"""Exercise the release interfaces outside a Git checkout."""

import hashlib
import importlib.util
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


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


def test_submission_bundles_only_the_reviewed_dependency(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "build_submission", ROOT / "scripts/build_submission.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    (source / "config").mkdir(parents=True)
    (source / "imindbench").mkdir()
    commit = "abcdef0" + "1" * 33
    config_name = "config/torch_brain_dependency.json"
    (source / "README.md").write_text(
        'python -m pip install "torch_brain @ git+https://example.org/upstream@commit"\n'
    )
    (source / "config/submission_files.json").write_text(
        json.dumps([config_name, "README.md"])
    )
    wheel = tmp_path / "torch_brain-0.2.1+gabcdef0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "torch_brain-0.2.1+gabcdef0.dist-info/METADATA",
            "Name: torch_brain\nVersion: 0.2.1+gabcdef0\n",
        )
    original = json.dumps(
        {
            "torch_brain_commit": commit,
            "torch_brain_artifact_sha256": hashlib.sha256(
                wheel.read_bytes()
            ).hexdigest(),
        }
    )
    (source / config_name).write_text(original)
    output = tmp_path / "bundle.tar.gz"
    module.build_submission(source, output, wheel)
    with tarfile.open(output) as archive:
        bundled = json.load(archive.extractfile("imindbench/" + config_name))
        assert (
            bundled["torch_brain_artifact_sha256"]
            == hashlib.sha256(wheel.read_bytes()).hexdigest()
        )
        assert (
            archive.extractfile("imindbench/vendor/" + wheel.name).read()
            == wheel.read_bytes()
        )
        assert (
            b"pip install vendor/torch_brain"
            in archive.extractfile("imindbench/README.md").read()
        )
    assert (source / config_name).read_text() == original
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("unexpected.txt", "Different wheel with same version")
    with pytest.raises(ValueError, match="SHA256 does not match"):
        module.build_submission(source, tmp_path / "different.tar.gz", wheel)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "torch_brain.dist-info/METADATA", "Name: torch_brain\nVersion: 0.0.0\n"
        )
    with pytest.raises(ValueError, match="reviewed clean source commit"):
        module.build_submission(source, tmp_path / "wrong.tar.gz", wheel)


def test_release_allowlist_excludes_private_verification_material(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "build_submission", ROOT / "scripts/build_submission.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "release.tar.gz"
    module.build_submission(ROOT, output)
    with tarfile.open(output) as archive:
        names = [member.name.removeprefix("imindbench/") for member in archive]
    assert not any(name.startswith(("provenance/", "artifacts/")) for name in names)
    assert not any("parity" in name or "reference.json" in name for name in names)
    assert "config/torch_brain_dependency.json" in names
    assert "scripts/build_submission.py" in names
    assert "scripts/run_experiments.sh" in names
    assert "imindbench/conf/experiment/paper_multistft/mlp.yaml" in names
    assert not any(name.startswith("imindbench/recipes/") for name in names)
