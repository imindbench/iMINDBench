"""Generate the exhaustive inventory of retained repository files."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = Path("provenance/surface/retained.tsv")


def classify_path(path: Path) -> tuple[str, str]:
    """Return the retention reason and operational reachability for ``path``."""
    source_path = path.as_posix()
    if source_path.startswith("LICENSE") or source_path == "THIRD_PARTY.md":
        return "license metadata", "distributed with the release"
    if source_path.endswith(".md"):
        return "release documentation", "public repository documentation"
    if source_path.endswith(".gitignore"):
        return "repository metadata", "controls public source selection"
    if source_path.startswith("artifacts/parity_reference/"):
        return "canonical parity reference", "consumed by scripts/parity_tools.py"
    if source_path.startswith("config/"):
        return "public validation configuration", "consumed by validation tooling"
    if source_path.startswith("imindbench/conf/"):
        return "runtime configuration", "composed by imindbench.run_eval"
    if source_path.startswith("imindbench/decodable_subject_sessions/"):
        return "stable evaluation artifact", "consumed by retained launchers"
    if source_path.startswith("imindbench/models/"):
        return "retained model implementation", "registered by imindbench.models"
    if source_path.startswith("imindbench/preprocessors/"):
        return (
            "retained preprocessing implementation",
            "registered by imindbench.preprocessors",
        )
    if source_path.startswith("imindbench/scripts/"):
        return "operational evaluation launcher", "packaged shell entrypoint"
    if source_path.startswith("imindbench/"):
        return "evaluation runtime", "imported by the imindbench package"
    if source_path == "scripts/parity_tools.py":
        return "parity validation tooling", "public dry-run and comparison entrypoint"
    if source_path.startswith("scripts/"):
        return "release validation tooling", "public validation entrypoint"
    if source_path.startswith("tests/"):
        return "verification and source identity", "exercised by the test suite"
    if source_path.startswith("provenance/"):
        return "release provenance", "audit and retained-surface evidence"
    if source_path in {"pyproject.toml", "environment.yml"}:
        return "build and environment metadata", "used for installation"
    return "release support file", "retained in the public repository"


def collect_paths() -> list[Path]:
    """Collect every tracked or untracked, non-ignored file that still exists."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    paths = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        path = Path(raw_path.decode("utf-8"))
        if not (REPOSITORY_ROOT / path).is_file():
            continue
        if "\t" in path.as_posix() or "\n" in path.as_posix():
            raise ValueError(f"TSV-incompatible repository path: {path!s}")
        paths.append(path)
    return sorted(paths, key=lambda candidate: candidate.as_posix())


def generate_manifest() -> str:
    """Return the complete retained-file manifest as deterministic TSV text."""
    lines = ["source_path\treason\treachability"]
    for path in collect_paths():
        reason, reachability = classify_path(path)
        lines.append(f"{path.as_posix()}\t{reason}\t{reachability}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the exhaustive retained-file manifest."""
    (REPOSITORY_ROOT / OUTPUT_PATH).write_text(
        generate_manifest(), encoding="utf-8", newline=""
    )


if __name__ == "__main__":
    main()
