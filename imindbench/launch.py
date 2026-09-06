"""Plan or execute recipe grids through the installed evaluation entrypoint."""

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

from omegaconf import OmegaConf

RECIPE_DIR = Path(__file__).resolve().parent / "recipes"
POPULATION_DIR = Path(__file__).resolve().parent / "decodable_subject_sessions"
IDENTITY_KEYS = {
    "paths",
    "dataset",
    "model",
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


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _names(value, name):
    if (
        not isinstance(value, list)
        or not value
        or not all(
            isinstance(item, str) and re.fullmatch(r"[\w-]+", item) for item in value
        )
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{name} must be a nonempty list of unique names")
    return value


def _slug(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[\w-]+", value):
        raise ValueError(f"{name} must be a simple name, not a path")
    return value


def _select(available, requested, name):
    if requested is None:
        return available
    if len(requested) != len(set(requested)) or set(requested) - set(available):
        raise ValueError(f"{name} must select unique entries from {available}")
    return [item for item in available if item in requested]


def _value(value):
    # Preserve historical sweep spellings (e.g. 0.50) as Hydra scalar literals.
    if isinstance(value, str) and re.fullmatch(r"[\w./:+-]+", value):
        return value
    return json.dumps(value, allow_nan=False)


def load_recipe(recipe_name, dataset):
    """Load and validate one dataset's recipe at the configuration boundary."""
    path = Path(recipe_name)
    if path.suffix != ".yaml":
        path = RECIPE_DIR / f"{_slug(recipe_name, 'recipe')}.yaml"
    recipe = _mapping(
        OmegaConf.to_container(OmegaConf.load(path), resolve=True), "recipe"
    )
    allowed = {
        "description",
        "target_set",
        "overrides",
        "datasets",
        "tasks",
        "decodable_rule",
        "device_overrides",
    }
    if set(recipe) - allowed:
        raise ValueError(f"Unknown recipe fields: {set(recipe) - allowed}")
    datasets = _mapping(recipe.get("datasets"), "datasets")
    if dataset not in datasets:
        raise ValueError(f"Recipe supports datasets: {list(datasets)}")
    selected = _mapping(datasets[dataset], "dataset recipe")
    allowed = {
        "dataset",
        "output_group",
        "preprocessor",
        "models",
        "regimes",
        "overrides",
        "output_subset",
        "sweep",
    }
    if set(selected) - allowed:
        raise ValueError(f"Unknown dataset recipe fields: {set(selected) - allowed}")
    for key in ("dataset", "output_group", "preprocessor"):
        _slug(selected.get(key), key)
    _names(selected.get("regimes"), "regimes")
    models = _mapping(selected.get("models"), "models")
    _names(list(models), "models")
    for model, settings in models.items():
        if set(_mapping(settings, model)) - {"preprocessor", "overrides"}:
            raise ValueError(f"{model} supports preprocessor and model overrides only")
        for key, value in _mapping(settings.get("overrides", {}), model).items():
            # Model-specific tuning must not replace grid identity or routing.
            if (
                not isinstance(key, str)
                or not re.fullmatch(r"(?:\+\+)?(?:model|dataset)\.[\w.]+", key)
                or key.lstrip("+") in IDENTITY_KEYS | {"dataset.subset_tier"}
                or (value is not None and type(value) not in (str, bool, int, float))
            ):
                raise ValueError(f"Invalid model-specific override: {key}")
            _value(value)
        if "preprocessor" in settings:
            _slug(settings["preprocessor"], "model preprocessor")
    shared = OmegaConf.to_container(
        OmegaConf.load(RECIPE_DIR / "datasets.yaml"), resolve=True
    )
    target_set = _slug(recipe.get("target_set"), "target_set")
    targets = shared["datasets"][dataset]["targets"][target_set]
    if (
        not isinstance(targets, list)
        or not targets
        or not all(
            isinstance(pair, list)
            and len(pair) == 2
            and all(type(v) is int and v >= 0 for v in pair)
            for pair in targets
        )
    ):
        raise ValueError("Targets must be nonnegative integer [subject, session] pairs")
    if len(targets) != len({tuple(pair) for pair in targets}):
        raise ValueError("Duplicate target pairs")
    tasks = _names(recipe.get("tasks", shared["tasks"]), "tasks")
    overrides = {
        **_mapping(recipe.get("overrides", {}), "overrides"),
        **_mapping(selected.get("overrides", {}), "dataset overrides"),
    }
    for key, value in overrides.items():
        if not isinstance(key, str) or not re.fullmatch(r"\+{0,2}[\w.]+", key):
            raise ValueError(f"Invalid override key: {key}")
        if key.lstrip("+") in IDENTITY_KEYS - {"dataset.subset_tier"}:
            raise ValueError(f"Recipe overrides cannot replace launcher field {key}")
        if value is not None and type(value) not in (str, bool, int, float):
            raise ValueError(f"Override {key} must be a scalar")
        _value(value)
    output_subset = selected.get("output_subset", False)
    if not isinstance(output_subset, bool):
        raise ValueError("output_subset must be boolean")
    if output_subset:
        _slug(overrides.get("dataset.subset_tier"), "dataset.subset_tier")
    sweep = _mapping(selected.get("sweep", {}), "sweep")
    for key, dimension in sweep.items():
        if (
            not isinstance(key, str)
            or not re.fullmatch(r"[\w.]+", key)
            or key in IDENTITY_KEYS
            or key in overrides
        ):
            raise ValueError(f"Sweep key collides with identity/overrides: {key}")
        dimension = _mapping(dimension, "sweep dimension")
        if set(dimension) != {"label", "values"}:
            raise ValueError("Sweep dimensions require label and values")
        _slug(dimension["label"], "sweep label")
        values = dimension["values"]
        if (
            not isinstance(values, list)
            or not values
            or not all(
                isinstance(v, str) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", v)
                for v in values
            )
            or len(set(values)) != len(values)
        ):
            raise ValueError("Sweep values must be unique numeric Hydra scalar strings")
    device_overrides = recipe.get("device_overrides", [])
    if not isinstance(device_overrides, list) or any(
        not isinstance(key, str)
        or not re.fullmatch(r"\+{0,2}preprocessor\.[\w.]+\.device", key)
        for key in device_overrides
    ):
        raise ValueError("device_overrides must name preprocessor device fields")
    rule = recipe.get("decodable_rule")
    if rule is not None:
        _slug(rule, "decodable_rule")
    return {
        **selected,
        "tasks": tasks,
        "targets": targets,
        "overrides": overrides,
        "sweep": sweep,
        "device_overrides": device_overrides,
        "decodable_rule": rule,
    }


def build_commands(args):
    """Expand the selected grid without creating output directories or running jobs."""
    recipe = load_recipe(args.recipe, args.dataset)
    output = args.output_root.expanduser().resolve()
    models = _select(list(recipe["models"]), args.model, "model")
    regimes = _select(recipe["regimes"], args.regime, "regime")
    tasks = _select(recipe["tasks"], args.task, "task")
    targets = {f"sub{s}_sess{t}": (s, t) for s, t in recipe["targets"]}
    selected_targets = _select(list(targets), args.target, "target")
    config_args = []
    if args.config_dir is not None:
        if not args.config_dir.is_dir():
            raise ValueError("config-dir must be an existing directory")
        config_args = ["--config-dir", str(args.config_dir.resolve())]
    _slug(args.paths, "paths")
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", args.device):
        raise ValueError("device must be cpu, cuda or cuda:<index>")
    extra = {}
    controlled = IDENTITY_KEYS | set(recipe["sweep"]) | set(recipe["device_overrides"])
    for override in args.overrides:
        key, separator, value = override.partition("=")
        if not separator or not re.fullmatch(r"\+{0,2}[\w.]+", key):
            raise ValueError("--set expects a Hydra key=value override")
        plain = key.lstrip("+")
        if plain in {k.lstrip("+") for k in controlled} or plain.startswith("hydra."):
            raise ValueError(f"Use recipe fields/selection flags to change {key}")
        if plain in extra:
            raise ValueError(f"Duplicate --set override: {key}")
        extra[plain] = (key, value)
    population = None
    decodable_dir = None
    if recipe["decodable_rule"] is not None:
        decodable_dir = (
            args.decodable_dir or POPULATION_DIR / recipe["decodable_rule"]
        ).resolve()
        population = json.loads((decodable_dir / f"{args.dataset}.json").read_text())
        population = _mapping(population.get("tasks"), "population tasks")
        for task in tasks:
            task_population = _mapping(population.get(task), f"population task {task}")
            _names(
                task_population.get("subject_sessions"), "population subject_sessions"
            )
    elif args.decodable_dir is not None:
        raise ValueError("decodable-dir requires a recipe with decodable_rule")
    sweep = recipe["sweep"]
    combinations = list(itertools.product(*(dim["values"] for dim in sweep.values())))
    commands = []
    run_paths = set()
    for model, regime, values, task, target in itertools.product(
        models, regimes, combinations, tasks, selected_targets
    ):
        if (
            population is not None
            and target not in population[task]["subject_sessions"]
        ):
            continue
        subject, session = targets[target]
        preprocessor = recipe["models"][model].get(
            "preprocessor", recipe["preprocessor"]
        )
        overrides = {
            "paths": args.paths,
            "dataset": recipe["dataset"],
            "dataset.regime": regime,
            "dataset.task": task,
            "dataset.test_subject": subject,
            "dataset.test_session": session,
            "model": model,
            "preprocessor": preprocessor,
            **recipe["overrides"],
            **recipe["models"][model].get("overrides", {}),
            **dict(zip(sweep, values, strict=True)),
            "wandb.enabled": False,
            "runtime.overwrite": False,
            "runtime.verbose": True,
        }
        if model != "logistic":
            overrides["model.device"] = args.device
        for key in recipe["device_overrides"]:
            overrides[key] = args.device
        if decodable_dir is not None:
            overrides["paths.decodable_subject_sessions_dir"] = str(decodable_dir)
        # Session-only policy does not apply to within-session/subject holdout.
        if regime not in {"hold-in-session", "hold-out-session"}:
            overrides["dataset.train_same_subject_only"] = False
        run_dir = output / recipe["output_group"]
        if recipe.get("output_subset", False):
            run_dir /= overrides["dataset.subset_tier"]
        run_dir /= f"{model}_{preprocessor}"
        if sweep:
            run_dir /= "_".join(
                dim["label"] + value.replace(".", "p")
                for dim, value in zip(sweep.values(), values, strict=True)
            )
        run_dir = run_dir / regime / task / target
        if run_dir in run_paths:
            raise ValueError(f"Duplicate output directory: {run_dir}")
        run_paths.add(run_dir)
        overrides["hydra.run.dir"] = str(run_dir)
        tokens = [
            f"{key}={_value(value)}"
            for key, value in overrides.items()
            if key.lstrip("+") not in extra
        ]
        tokens.extend(f"{key}={value}" for key, value in extra.values())
        command = [sys.executable, "-m", "imindbench.run_eval", *config_args, *tokens]
        commands.append(
            {
                "command": command,
                "run_dir": str(run_dir),
                "result": str(
                    run_dir / f"population_btbank{subject}_{session}_{task}.json"
                ),
            }
        )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        commands = commands[: args.limit]
    if not commands:
        raise ValueError(
            "No evaluations remain after recipe selections/population filtering"
        )
    return commands


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
    """Return the public launcher argument parser."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--recipe", required=True, help="Packaged recipe name or custom .yaml file"
    )
    result.add_argument(
        "--dataset", required=True, help="Canonical dataset/provider name"
    )
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--paths", default="example", help="Hydra paths group")
    result.add_argument("--config-dir", type=Path)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--model", action="append")
    result.add_argument("--regime", action="append")
    result.add_argument("--task", action="append")
    result.add_argument("--target", action="append", help="Select sub<S>_sess<T>")
    result.add_argument("--decodable-dir", type=Path)
    result.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    result.add_argument("--limit", type=int)
    result.add_argument(
        "--count", action="store_true", help="Print only the planned evaluation count"
    )
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
    except (ValueError, KeyError, OSError) as exc:
        print(f"Grid error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
