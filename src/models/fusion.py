"""LightGBM and MLP fusion heads for combining branch features."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import joblib
import lightgbm as lgb
from lightgbm.basic import LightGBMError
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LightGBM fusion
# ---------------------------------------------------------------------------

class LGBMFusion:
    """LightGBM binary classifier operating on concatenated branch features.

    Usage:
        fusion = LGBMFusion(params)
        fusion.fit(X_train, y_train, X_val, y_val)
        scores = fusion.predict(X_test)   # fraud probabilities [0,1]
        fusion.save("checkpoints/lgbm_fusion.pkl")
    """

    def __init__(self, params: dict | None = None):
        self.params = params or {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": 2000,
            "learning_rate": 0.02,
            "num_leaves": 64,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "device": "cuda",
            "verbosity": -1,
        }
        self.model: lgb.LGBMClassifier | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        eval_set = [(X_val, y_val)] if X_val is not None else None
        callbacks = [lgb.early_stopping(50, verbose=True), lgb.log_evaluation(100)]
        fit_callbacks = callbacks if eval_set else [lgb.log_evaluation(100)]

        try:
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=fit_callbacks,
            )
        except LightGBMError as exc:
            device = self.params.get("device") or self.params.get("device_type")
            if device in {"gpu", "cuda"} and (
                "GPU Tree Learner was not enabled" in str(exc)
                or "CUDA Tree Learner was not enabled" in str(exc)
            ):
                print(f"LightGBM {device} build unavailable; retrying fusion training on CPU.")
                cpu_params = dict(self.params)
                cpu_params["device"] = "cpu"
                cpu_params.pop("device_type", None)
                self.params = cpu_params
                self.model = lgb.LGBMClassifier(**self.params)
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    callbacks=fit_callbacks,
                )
            else:
                raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model, path)
        print(f"Saved LightGBM model → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LGBMFusion":
        obj = cls()
        obj.model = joblib.load(path)
        return obj

    def feature_importance(self, feature_names: list[str] | None = None) -> dict:
        if self.model is None:
            return {}
        imp = self.model.feature_importances_
        names = feature_names or [f"f{i}" for i in range(len(imp))]
        return dict(sorted(zip(names, imp), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# MLP fusion (PyTorch alternative)
# ---------------------------------------------------------------------------

class MLPFusion(nn.Module):
    """Small MLP head for fusing concatenated branch features.

    Output: scalar sigmoid score per sample (fraud probability).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = (512, 256, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)          # (B,) raw logits


# ---------------------------------------------------------------------------
# Feature assembler: metadata → float vector
# ---------------------------------------------------------------------------

DOC_TYPE_VOCAB: dict[str, int] = {}   # populated at runtime


def encode_metadata(
    is_digital: int | None,
    doc_type: str | None,
    vocab: dict[str, int] | None = None,
) -> np.ndarray:
    """Encode a single sample's metadata into a float32 vector.

    Returns a 2-element vector: [is_digital_flag, doc_type_freq_encoded].
    is_digital: 1=digital, 0=recaptured, -1=unknown.
    doc_type: e.g. "USA/DL". Unknown types map to index 0.
    """
    vocab = vocab or DOC_TYPE_VOCAB
    dig_feat = float(is_digital) if is_digital is not None else -1.0
    type_feat = float(vocab.get(doc_type or "", 0))
    return np.array([dig_feat, type_feat], dtype=np.float32)
