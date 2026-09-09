"""Exercise native thread caps in fresh processes with small initial pools."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("initial_blas_threads", [1, 2, 8])
def test_sklearn_thread_cap_preserves_smaller_pools(initial_blas_threads):
    # BLAS reads its initial pool size at import time. A subprocess also contains
    # the native crash that motivated this regression test.
    code = r"""
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import numpy as np
from omegaconf import OmegaConf
from threadpoolctl import threadpool_info
from imindbench.models.logistic_model import LogisticModel
from imindbench.sklearn_runner import SKLearnRunner

def pool_sizes():
    return {p['filepath']: p['num_threads'] for p in threadpool_info()
            if p['num_threads'] is not None}

X = np.array([[i % 2] * 8 for i in range(8)], dtype=np.float32)
y = np.arange(8) % 2
model_cfg = OmegaConf.create({'max_iter': 10000, 'tol': 1e-4, 'random_state': 42})
original = pool_sizes()
reference = LogisticModel(model_cfg).fit(X, y)

class ObservedLogistic(LogisticModel):
    fail = False

    def fit(self, *args, **kwargs):
        active = pool_sizes()
        assert active == {path: min(4, count) for path, count in original.items()}
        if self.fail:
            raise RuntimeError('intentional fit failure')
        return super().fit(*args, **kwargs)

runner = SKLearnRunner(OmegaConf.create({'runtime': {'sklearn_num_threads': 4}}))
model = ObservedLogistic(model_cfg)
result = runner.run_fold(model, X, y, X, y)
assert result['test_accuracy'] == 1.0
np.testing.assert_allclose(model.predict_proba(X), reference.predict_proba(X))
assert pool_sizes() == original

model.fail = True
try:
    runner.run_fold(model, X, y, X, y)
except RuntimeError as exc:
    assert str(exc) == 'intentional fit failure'
else:
    raise AssertionError('expected fit to fail')
assert pool_sizes() == original
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        env={
            **os.environ,
            "OPENBLAS_NUM_THREADS": str(initial_blas_threads),
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": str(initial_blas_threads),
        },
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
