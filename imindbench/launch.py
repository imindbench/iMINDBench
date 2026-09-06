"""Preview or execute an evaluation grid using ordinary Hydra configurations."""

import argparse
import fcntl
import hashlib
import itertools
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import OmegaConf

CONF_DIR = Path(__file__).resolve().parent / "conf"
POPULATION_DIR = Path(__file__).resolve().parent / "decodable_subject_sessions"
IDENTITY_KEYS = {
    "paths",
    "dataset",
    "model",
    "experiment",
    "model.name",
    "preprocessor",
    "dataset.provider",
    "dataset.regime",
    "dataset.task",
    "dataset.test_subject",
    "dataset.test_session",
    "dataset.subset_tier",
    "model.device",
    "hydra.run.dir",
    "runtime.overwrite",
    "paths.decodable_subject_sessions_dir",
}


def _validate_name(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[\w-]+", value):
        raise ValueError(f"{name} must be a simple name")


def _validate_names(value, name, allow_empty=False):
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or any(not isinstance(v, str) or not re.fullmatch(r"[\w-]+", v) for v in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{name} must contain unique names")


def _select(available, requested, name):
    if requested is None:
        return available
    _validate_names(requested, name)
    if set(requested) - set(available):
        raise ValueError(f"{name} must select entries from {available}")
    return requested


def _value(value):
    if isinstance(value, str) and re.fullmatch(r"[\w./:+-]+", value):
        return value
    return json.dumps(value, allow_nan=False)


def _key_value(override):
    key, separator, value = override.partition("=")
    if not separator or not re.fullmatch(r"\+{0,2}[\w.]+", key):
        raise ValueError("Overrides must be Hydra key=value assignments")
    if key.lstrip("+") in IDENTITY_KEYS or key.lstrip("+").startswith("hydra."):
        raise ValueError(f"Use selection flags or experiment configs to change {key}")
    return key, value


def build_commands(args):
    """Expand one model's task/target/sweep grid without creating any outputs."""
    for name in (
        "dataset",
        "model",
        "preprocessor",
        "paths",
        "population",
        "output_group",
    ):
        _validate_name(getattr(args, name), name)
    if args.experiment is not None:
        if not re.fullmatch(r"[\w-]+(?:/[\w-]+)*", args.experiment):
            raise ValueError("experiment must name a Hydra config group entry")
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", args.device):
        raise ValueError("device must be cpu, cuda or cuda:<index>")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    config_args = []
    searchpath = []
    if args.config_dir is not None:
        config_dir = args.config_dir.expanduser().resolve()
        if not config_dir.is_dir():
            raise ValueError("config-dir must be an existing directory")
        config_args = ["--config-dir", str(config_dir)]
        searchpath = [f"hydra.searchpath={json.dumps(['file://' + str(config_dir)])}"]
    base = {
        "paths": args.paths,
        "dataset": args.dataset,
        "model": args.model,
        "preprocessor": args.preprocessor,
        "dataset.regime": args.regime,
        "wandb.enabled": False,
        "runtime.overwrite": False,
        "runtime.verbose": True,
    }
    if args.experiment is not None:
        base["experiment"] = args.experiment
    if args.subset is not None:
        _validate_name(args.subset, "subset")
        base["dataset.subset_tier"] = args.subset
    if args.model != "logistic":
        base["model.device"] = args.device
    if args.regime not in {"hold-in-session", "hold-out-session"}:
        base["dataset.train_same_subject_only"] = False
    decodable_dir = args.decodable_dir
    if decodable_dir is None and args.decodable_rule is not None:
        _validate_name(args.decodable_rule, "decodable-rule")
        decodable_dir = POPULATION_DIR / args.decodable_rule
    if decodable_dir is not None:
        decodable_dir = decodable_dir.expanduser().resolve()
        base["paths.decodable_subject_sessions_dir"] = str(decodable_dir)
    # CLI tuning comes after the experiment config; repeated --set uses the last value.
    extra = {}
    for override in args.overrides:
        key, value = _key_value(override)
        extra[key.lstrip("+")] = (key, value)
    sweeps = {}
    for dimension in args.sweep:
        key, value = _key_value(dimension)
        values = value.split(",")
        if (
            key.lstrip("+") in extra
            or key in sweeps
            or not re.fullmatch(r"[\w.]+", key)
            or not all(re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", v) for v in values)
            or len(values) != len(set(values))
        ):
            raise ValueError(
                "Sweep keys must be unique and separate from --set; values must be unique numeric scalars"
            )
        sweeps[key] = values
    tokens = [
        f"{key}={_value(value)}" for key, value in base.items() if key not in extra
    ]
    tokens += [f"{key}={value}" for key, value in extra.values()]
    # Ask Hydra for provider identity instead of inferring it from a config filename.
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.1"):
        cfg = compose(config_name="config", overrides=[*tokens, *searchpath])
    provider = cfg.dataset.provider
    _validate_name(provider, "dataset.provider")
    catalog = OmegaConf.to_container(
        OmegaConf.load(CONF_DIR / "population/catalog.yaml"), resolve=True
    )
    _validate_names(catalog["tasks"], "tasks")
    tasks = _select(catalog["tasks"], args.task, "task")
    pairs = catalog["datasets"][provider]["targets"][args.population]
    if (
        not isinstance(pairs, list)
        or not pairs
        or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(type(v) is not int or v < 0 for v in pair)
            for pair in pairs
        )
        or len({tuple(pair) for pair in pairs}) != len(pairs)
    ):
        raise ValueError(
            "Population targets must be unique nonnegative [subject, session] pairs"
        )
    targets = {f"sub{s}_sess{t}": (s, t) for s, t in pairs}
    selected_targets = _select(list(targets), args.target, "target")
    population = None
    if decodable_dir is not None:
        population = json.loads((decodable_dir / f"{provider}.json").read_text())[
            "tasks"
        ]
        if not isinstance(population, dict):
            raise ValueError("Decodable population tasks must be a mapping")
        for task in tasks:
            if (
                not isinstance(population.get(task), dict)
                or "subject_sessions" not in population[task]
            ):
                raise ValueError(
                    f"Decodable population must list subject_sessions for {task}"
                )
            # A task with no decodable recordings contributes zero jobs.
            _validate_names(
                population[task]["subject_sessions"],
                "decodable targets",
                allow_empty=True,
            )
    commands = []
    output = args.output_root.expanduser().resolve()
    for values, task, target in itertools.product(
        itertools.product(*sweeps.values()), tasks, selected_targets
    ):
        if (
            population is not None
            and target not in population[task]["subject_sessions"]
        ):
            continue
        subject, session = targets[target]
        run_dir = output / args.output_group / f"{args.model}_{args.preprocessor}"
        if args.subset is not None:
            run_dir /= args.subset
        if sweeps:
            run_dir /= "_".join(
                key.replace(".", "-") + "=" + value
                for key, value in zip(sweeps, values, strict=True)
            )
        run_dir = run_dir / args.regime / task / target
        job_tokens = [
            *tokens,
            *(f"{key}={value}" for key, value in zip(sweeps, values, strict=True)),
            f"dataset.task={task}",
            f"dataset.test_subject={subject}",
            f"dataset.test_session={session}",
            f"hydra.run.dir={_value(str(run_dir))}",
        ]
        commands.append(
            {
                "command": [
                    sys.executable,
                    "-m",
                    "imindbench.run_eval",
                    *config_args,
                    *job_tokens,
                ],
                "run_dir": str(run_dir),
                "result": str(
                    run_dir / f"population_btbank{subject}_{session}_{task}.json"
                ),
            }
        )
    if not commands:
        raise ValueError("No evaluations remain after selections/population filtering")
    return commands[: args.limit] if args.limit is not None else commands


def _completion_matches(job):
    result = Path(job["result"])
    marker = Path(job["run_dir"]) / "completed.sha256"
    return (
        result.is_file()
        and marker.is_file()
        and (
            hashlib.sha256(result.read_bytes()).hexdigest()
            == marker.read_text().strip()
        )
    )


def execute_commands(commands, output_root, resume=False):
    """Run serially, with exclusive ownership and explicit same-command resume."""
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".grid.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another grid is active in this output root") from exc
        # Check every destination before launching anything. Existing results from
        # older shell scripts must use a fresh root, not provenance-free resume.
        for job in commands:
            directory = Path(job["run_dir"])
            if directory.exists() and any(directory.iterdir()):
                record = directory / "launch.json"
                if (
                    not resume
                    or not record.is_file()
                    or json.loads(record.read_text()) != job["command"]
                ):
                    raise ValueError(
                        f"Existing run requires --resume and the same launch.json: {directory}"
                    )
            if Path(job["result"]).exists() and not _completion_matches(job):
                raise ValueError(
                    f"Unverified or changed result; inspect it and use a fresh output root: {directory}"
                )
        failures = 0
        for job in commands:
            directory = Path(job["run_dir"])
            if resume and _completion_matches(job):
                print(f"Skipping completed run: {directory}")
                continue
            directory.mkdir(parents=True, exist_ok=True)
            record = directory / "launch.json"
            if not record.exists():
                with record.open("x") as stream:
                    json.dump(job["command"], stream, indent=2)
                    stream.write("\n")
            print(shlex.join(job["command"]), flush=True)
            with (directory / "launcher.log").open("a") as log:
                result = subprocess.run(
                    job["command"], stdout=log, stderr=subprocess.STDOUT
                )
            if result.returncode or not Path(job["result"]).is_file():
                failures += 1
                print(
                    f"Failed evaluation; see {directory / 'launcher.log'}",
                    file=sys.stderr,
                )
            else:
                payload = Path(job["result"]).read_bytes()
                try:
                    json.loads(payload)
                except (ValueError, UnicodeDecodeError):
                    failures += 1
                    print(f"Invalid result JSON: {job['result']}", file=sys.stderr)
                    continue
                marker = directory / "completed.sha256.tmp"
                marker.write_text(hashlib.sha256(payload).hexdigest() + "\n")
                marker.replace(directory / "completed.sha256")
        return failures


def parser():
    """Return the shared grid execution interface used by the shell scripts."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True, help="Hydra dataset config")
    result.add_argument("--model", required=True, help="Hydra model config")
    result.add_argument(
        "--preprocessor", required=True, help="Hydra preprocessor config"
    )
    result.add_argument("--experiment", help="Optional Hydra experiment config")
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--output-group", default="evaluations")
    result.add_argument("--paths", default="example")
    result.add_argument("--config-dir", type=Path)
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--regime",
        default="within-session",
        choices=[
            "within-session",
            "hold-in-session",
            "hold-out-session",
            "hold-out-subject",
        ],
    )
    result.add_argument(
        "--population", default="all", help="Named task/target catalog population"
    )
    result.add_argument("--subset", help="Dataset subset tier")
    result.add_argument(
        "--task",
        nargs="+",
        help="Select tasks; the last --task replaces earlier selections",
    )
    result.add_argument(
        "--target",
        nargs="+",
        help="Select sub<S>_sess<T>; the last --target replaces earlier selections",
    )
    result.add_argument(
        "--decodable-dir",
        type=Path,
        help="Custom population directory; takes precedence over --decodable-rule",
    )
    result.add_argument(
        "--decodable-rule", help="Packaged decodable population directory"
    )
    result.add_argument("--sweep", action="append", default=[], metavar="KEY=V1,V2")
    result.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    result.add_argument("--limit", type=int)
    result.add_argument("--count", action="store_true")
    result.add_argument(
        "--execute", action="store_true", help="Execute; default only prints commands"
    )
    result.add_argument("--resume", action="store_true")
    return result


def main():
    args = parser().parse_args()
    try:
        if args.count and args.execute:
            raise ValueError("--count cannot be combined with --execute")
        if args.resume and not args.execute:
            raise ValueError("--resume requires --execute")
        commands = build_commands(args)
        if args.count:
            print(len(commands))
        elif args.execute:
            failures = execute_commands(
                commands, args.output_root.expanduser().resolve(), args.resume
            )
            print(f"{len(commands)} planned evaluations; {failures} failed")
            return int(failures > 0)
        else:
            for job in commands:
                print(shlex.join(job["command"]))
            print(
                f"{len(commands)} evaluations; dry run (nothing executed)",
                file=sys.stderr,
            )
    except (ValueError, KeyError, OSError, HydraException) as exc:
        print(f"Grid error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
