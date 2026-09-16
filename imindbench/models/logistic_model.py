"""
Logistic Regression model wrapper for sklearn.
"""

import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression

from . import register_model
from .base_model import BaseModel


@register_model("logistic")
class LogisticModel(BaseModel):
    """Wrapper for sklearn LogisticRegression."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = LogisticRegression(
            C=cfg.get("C", 1.0),
            penalty=cfg.get("penalty", "l2"),
            solver=cfg.get("solver", "lbfgs"),
            class_weight=cfg.get("class_weight", None),
            max_iter=cfg.get("max_iter", 10000),
            tol=cfg.get("tol", 1e-4),
            random_state=cfg.get("random_state", 42),
        )

    def prepare_batch(self, batch, **kwargs):
        """Prepare one collated batch for sklearn logistic input."""
        _ = kwargs
        x = np.asarray(batch["x"], dtype=np.float32)
        if x.ndim < 2:
            raise ValueError(
                "LogisticModel expects 'x' with shape (batch, ...), got "
                f"{tuple(x.shape)}."
            )
        x = x.reshape(x.shape[0], -1)

        y = np.asarray(batch["y"])
        if y.ndim != 1:
            y = y.reshape(-1)
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                "prepare_batch expected matching batch dimensions for 'x' and 'y', "
                f"got {x.shape[0]} and {y.shape[0]}."
            )

        out = dict(batch)
        out["x"] = x
        out["y"] = y
        return out

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the logistic regression model."""
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)

    def score(self, X, y):
        """Compute accuracy score."""
        return self.model.score(X, y)
