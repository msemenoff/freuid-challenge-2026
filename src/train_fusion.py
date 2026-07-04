"""Train the LightGBM fusion head on pre-extracted feature caches.

Loads features from features/train/ and features/val/ (or uses a fold split
on features/all/), trains LightGBM, evaluates FREUID Score, and saves the
fitted model + threshold to checkpoints/.

Usage:
    python src/train_fusion.py
    python src/train_fusion.py --optuna_trials 50    # Optuna hyperparameter search
    python src/train_fusion.py --fit_all             # fit final model on all cached labels
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.data import load_train_dataframe
from src.utils.seed import seed_everything
from src.utils.metrics import freuid_score, find_best_threshold
from src.models.fusion import LGBMFusion
from src.dataset import make_folds


def load_features(split_dir: Path) -> dict[str, np.ndarray]:
    data = {}
    for fname in ["global_feats.npy", "dinov2_feats.npy", "physical_feats.npy",
                  "metadata.npy", "labels.npy", "ids.npy"]:
        fpath = split_dir / fname
        if fpath.exists():
            data[fname.replace(".npy", "")] = np.load(fpath, allow_pickle=True)
    return data


def _resolve_feature_keys(feature_sources: list[str] | tuple[str, ...] | None) -> list[str]:
    if not feature_sources:
        return ["global_feats", "dinov2_feats", "physical_feats", "metadata"]

    alias_map = {
        "global": "global_feats",
        "global_feats": "global_feats",
        "dinov2": "dinov2_feats",
        "dinov2_cls": "dinov2_feats",
        "dinov2_patch": "dinov2_feats",
        "dinov2_feats": "dinov2_feats",
        "physical": "physical_feats",
        "physical_feats": "physical_feats",
        "metadata": "metadata",
    }

    resolved = []
    for source in feature_sources:
        key = alias_map.get(source)
        if key and key not in resolved:
            resolved.append(key)
    return resolved


def assemble_X(data: dict, feature_sources: list[str] | tuple[str, ...] | None = None) -> np.ndarray:
    parts = []
    for key in _resolve_feature_keys(feature_sources):
        if key in data:
            parts.append(data[key].astype(np.float32))
    return np.concatenate(parts, axis=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--train_dir", default=None, help="features/train dir")
    p.add_argument("--val_dir", default=None, help="features/val dir")
    p.add_argument("--all_dir", default=None,
                   help="features/all dir (used with fold splitting)")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--fit_all", action="store_true",
                   help="Train on all labeled cached features without a validation split")
    p.add_argument("--optuna_trials", type=int, default=0,
                   help="Run Optuna HPO with N trials (0 = use config defaults)")
    p.add_argument("--out", default=None, help="Override output path for model")
    return p.parse_args()


def train_and_eval(
    X_train, y_train, X_val, y_val,
    params: dict,
    ckpt_path: Path,
    bpcer_target: float = 0.01,
) -> tuple[float, float]:
    fusion = LGBMFusion(params)
    fusion.fit(X_train, y_train, X_val, y_val)
    scores = fusion.predict(X_val)
    metrics = freuid_score(y_val, scores, bpcer_target)
    thresh, _ = find_best_threshold(y_val, scores, bpcer_target)
    print(
        f"  AuDET={metrics['audet']:.4f}  "
        f"APCER@1%BPCER={metrics['apcer_bpcer1']:.4f}  "
        f"FREUID={metrics['freuid']:.4f}  "
        f"threshold={thresh:.4f}"
    )
    fusion.save(ckpt_path)
    # Save threshold alongside model
    thresh_path = ckpt_path.with_suffix(".threshold.json")
    with open(thresh_path, "w") as f:
        json.dump({"threshold": thresh, **metrics}, f, indent=2)
    return metrics["freuid"], thresh


def train_full(X_train, y_train, params: dict, ckpt_path: Path) -> None:
    fusion = LGBMFusion(params)
    fusion.fit(X_train, y_train)
    fusion.save(ckpt_path)


def run_optuna(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int,
    bpcer_target: float,
    device: str,
) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "device": device,
        }
        import lightgbm as lgb
        import io, contextlib
        model = lgb.LGBMClassifier(**params)
        with contextlib.redirect_stdout(io.StringIO()):
            model.fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]
        return freuid_score(y_val, scores, bpcer_target)["freuid"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best FREUID Score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.data.seed)

    features_root = Path(cfg.paths.features_dir)
    ckpt_dir = Path(cfg.paths.checkpoints_dir)
    ckpt_dir.mkdir(exist_ok=True)

    bpcer_target = cfg.evaluation.bpcer_target
    fold = args.fold if args.fold is not None else cfg.data.fold

    # ---- Determine feature directories ----
    if args.fit_all:
        all_dir = Path(args.all_dir) if args.all_dir else features_root / "all"
        data = load_features(all_dir)
        X_train = assemble_X(data, cfg.fusion.feature_sources)
        y_train = data["labels"]
        X_val = y_val = None
    elif args.all_dir or (not args.train_dir and not args.val_dir):
        # Single features/all split – subdivide using the configured fold strategy
        all_dir = Path(args.all_dir) if args.all_dir else features_root / "all"
        data = load_features(all_dir)
        labels = data["labels"]
        train_df = load_train_dataframe(cfg)
        train_df = make_folds(
            train_df,
            n_folds=cfg.data.n_folds,
            seed=cfg.data.seed,
            label_col=cfg.data.label_col,
            strategy=getattr(cfg.data, "fold_strategy", "stratified"),
            group_col=getattr(cfg.data, "fold_group_col", "type"),
        )
        id_to_fold = dict(zip(train_df[cfg.data.id_col].astype(str), train_df["fold"].astype(int)))
        ids = data["ids"].astype(str)
        folds = np.array([id_to_fold[sample_id] for sample_id in ids], dtype=np.int64)
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]
        X_all = assemble_X(data, cfg.fusion.feature_sources)
        X_train, y_train = X_all[train_idx], labels[train_idx]
        X_val, y_val = X_all[val_idx], labels[val_idx]
    else:
        train_dir = Path(args.train_dir) if args.train_dir else features_root / "train"
        val_dir = Path(args.val_dir) if args.val_dir else features_root / "val"
        train_data = load_features(train_dir)
        val_data = load_features(val_dir)
        X_train = assemble_X(train_data, cfg.fusion.feature_sources)
        y_train = train_data["labels"]
        X_val = assemble_X(val_data, cfg.fusion.feature_sources)
        y_val = val_data["labels"]

    print(f"X_train: {X_train.shape}  positives: {y_train.sum():.0f}/{len(y_train)}")
    if X_val is not None and y_val is not None:
        print(f"X_val:   {X_val.shape}    positives: {y_val.sum():.0f}/{len(y_val)}")

    # ---- Optional Optuna HPO ----
    if args.optuna_trials > 0:
        if args.fit_all:
            raise ValueError("--optuna_trials requires a validation split; do not combine with --fit_all")
        best_params = run_optuna(X_train, y_train, X_val, y_val,
                                 args.optuna_trials, bpcer_target,
                                 device=cfg.fusion.lgbm.device)
        lgbm_params = {
            "objective": "binary", "metric": "auc", "verbosity": -1,
            "device": cfg.fusion.lgbm.device, **best_params,
        }
    else:
        lgbm_params = dict(cfg.fusion.lgbm)

    # ---- Train ----
    if args.fit_all:
        ckpt_path = Path(args.out) if args.out else ckpt_dir / "lgbm_fusion_full.pkl"
        print(f"\nTraining LightGBM fusion on all cached labels → {ckpt_path}")
        train_full(X_train, y_train, lgbm_params, ckpt_path)
    else:
        fold_tag = f"fold{fold}"
        ckpt_path = Path(args.out) if args.out else ckpt_dir / f"lgbm_fusion_{fold_tag}.pkl"
        print(f"\nTraining LightGBM fusion → {ckpt_path}")
        train_and_eval(X_train, y_train, X_val, y_val, lgbm_params, ckpt_path, bpcer_target)


if __name__ == "__main__":
    main()
