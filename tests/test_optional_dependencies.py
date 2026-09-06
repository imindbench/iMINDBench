"""Keep baseline imports usable when specialized model dependencies are absent."""

import subprocess
import sys


def test_baseline_config_and_model_work_without_specialized_dependencies(tmp_path):
    # A fresh process prevents the parent test suite's model imports from masking
    # a dependency leak. Only specialized third-party packages are hidden.
    code = r"""
import importlib.abc
import sys

class MissingExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"xformers", "mup", "jaxtyping", "warmup_scheduler"}:
            raise ModuleNotFoundError("Optional dependency unavailable", name=fullname)

sys.meta_path.insert(0, MissingExtras())
from hydra import compose, initialize_config_module
from imindbench.models import build_model
from imindbench.schedulers import build_scheduler
from imindbench.utils.pipeline_contracts import validate_eval_config

with initialize_config_module(config_module="imindbench.conf", version_base="1.1"):
    cfg = compose(config_name="config", overrides=[
        "paths=example", "model=logistic", "preprocessor=laplacian_multi_stft_2048Hz"
    ])
    validate_eval_config(cfg)
    assert build_model(cfg.model).__class__.__name__ == "LogisticModel"
    for model, extra in [("barista", "barista,warmup"), ("diver", "diver")]:
        selected = compose(config_name="config", overrides=[
            "paths=example", "model=" + model, "preprocessor=laplacian_wav_2048Hz"
        ])
        if model == "diver":
            selected.model.upstream_ckpt = "/unused/checkpoint.pt"
            selected.model.device = "cpu"
        try:
            instance = build_model(selected.model)
            if model == "diver":
                instance.build_model((1, 1, 1000), 2, device="cpu")
        except ImportError as exc:
            assert "imindbench[" + extra + "]" in str(exc), str(exc)
        else:
            raise AssertionError("Missing extra must fail before checkpoint loading")
    try:
        build_scheduler({"name": "ramp_up"}, None)
    except ModuleNotFoundError as exc:
        assert "imindbench[warmup]" in str(exc), str(exc)
    else:
        raise AssertionError("Missing scheduler extra must fail at selection")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
