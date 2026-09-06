"""Build a reproducible source archive from the reviewed submission allowlist."""

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path


def build_submission(source_root: Path, output: Path) -> None:
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
        contents[name] = payload

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
    args = parser.parse_args()
    build_submission(Path(__file__).resolve().parents[1], args.output)


if __name__ == "__main__":
    main()
