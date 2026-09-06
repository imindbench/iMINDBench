"""Build a reproducible source archive from the reviewed submission allowlist."""

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
import zipfile
from email import message_from_bytes
from pathlib import Path


def build_submission(
    source_root: Path, output: Path, torch_brain_wheel: Path | None = None
) -> None:
    """Export reviewed files without Git history, local state or personal paths."""
    source_root = source_root.resolve()
    manifest = source_root / "config/submission_files.json"
    files = json.loads(manifest.read_text())
    if (
        not isinstance(files, list)
        or not files
        or not all(isinstance(name, str) for name in files)
    ):
        raise ValueError("Submission allowlist must be a nonempty list of paths")
    if len(files) != len(set(files)):
        raise ValueError("Submission allowlist contains duplicate paths")

    contents = {}
    for name in sorted(files):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid submission path: {name}")
        path = source_root / relative
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError(f"Submission files must not traverse symlinks: {name}")
        payload = path.read_bytes()
        text = payload.decode("utf-8")
        # These are local identity leaks, not third-party attribution notices.
        if re.search(r"/(?:home|Users)/[^\s/]+|git[@][\w.-]+:", text):
            raise ValueError(f"Personal path or identity in submission file: {name}")
        if name == "tests/launcher_reference.json":
            # Keep command evidence; omit incidental private migration identity.
            reference = json.loads(payload)
            reference.pop("source_commit", None)
            payload = (json.dumps(reference, indent=2) + "\n").encode()
        contents[name] = payload

    if torch_brain_wheel is not None:
        if torch_brain_wheel.suffix != ".whl":
            raise ValueError("torch-brain-wheel must be a wheel file")
        wheel_bytes = torch_brain_wheel.read_bytes()
        manifest_name = "config/brainsets_smoke_manifest.json"
        dependency = json.loads(contents[manifest_name])
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as wheel:
            metadata_files = [
                name
                for name in wheel.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_files) != 1:
                raise ValueError("Wheel must have exactly one package metadata record")
            metadata = message_from_bytes(wheel.read(metadata_files[0]))
            version = metadata.get("Version", "")
            commit = re.search(r"(?:^|[.+])g([0-9a-f]{7,40})(?:$|[.])", version)
            if (
                metadata.get("Name", "").replace("-", "_") != "torch_brain"
                or commit is None
                or not dependency["torch_brain_commit"].startswith(commit[1])
                or f"g{commit[1]}.d" in version
            ):
                raise ValueError(
                    "TorchBrain wheel must identify the reviewed clean source commit"
                )
            for name in wheel.namelist():
                if name.endswith(
                    (".py", ".md", ".toml", ".yaml", ".json", "/METADATA")
                ):
                    if re.search(
                        rb"/(?:home|Users)/[^\s/]+|git[@][\w.-]+:", wheel.read(name)
                    ):
                        raise ValueError(f"Personal path in TorchBrain wheel: {name}")
        digest = hashlib.sha256(wheel_bytes).hexdigest()
        if digest != dependency["torch_brain_artifact_sha256"]:
            raise ValueError(
                "TorchBrain wheel SHA256 does not match the reviewed artifact"
            )
        wheel_name = f"vendor/{torch_brain_wheel.name}"
        contents[wheel_name] = wheel_bytes
        readme_name = "README.md"
        readme, substitutions = re.subn(
            r'^python -m pip install "torch_brain @ git\+[^\n]+$',
            f"python -m pip install {wheel_name}",
            contents[readme_name].decode(),
            flags=re.MULTILINE,
        )
        if substitutions != 1:
            raise ValueError(
                "Could not replace the single TorchBrain install step in the quickstart"
            )
        contents[readme_name] = readme.encode()
        contents["BUNDLE.md"] = (
            "# Bundled TorchBrain dependency\n\n"
            f"Install `{wheel_name}` using the included quickstart.\n"
            f"SHA256: `{digest}`.\n\n"
            "The source-validation manifest pins this exact wheel. Required upstream\n"
            "attribution and source fingerprints are retained. Inspect the final\n"
            "archive for submission-specific identifying information before sharing.\n"
            "This bundle supplies code, not data, checkpoints or the full environment.\n"
        ).encode()

    contents["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
        for name, payload in contents.items()
    ).encode()
    # Fixed archive metadata excludes the local account and build time.
    archive_bytes = io.BytesIO()
    with gzip.GzipFile(
        fileobj=archive_bytes, mode="wb", filename="", mtime=0
    ) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as archive:
            for name, payload in contents.items():
                entry = tarfile.TarInfo(f"imindbench/{name}")
                entry.size = len(payload)
                entry.mode = 0o755 if name.endswith(".sh") else 0o644
                archive.addfile(entry, io.BytesIO(payload))
    # Do not replace an already reviewed artifact.
    with output.open("xb") as stream:
        stream.write(archive_bytes.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--torch-brain-wheel",
        type=Path,
        help="Bundle a wheel from the reviewed dependency commit",
    )
    args = parser.parse_args()
    build_submission(
        Path(__file__).resolve().parents[1], args.output, args.torch_brain_wheel
    )


if __name__ == "__main__":
    main()
