"""Create a scored submission from cached features and a trained fusion model."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.data import load_public_test_dataframe, load_train_dataframe
from src.models.fusion import LGBMFusion
from src.train_fusion import assemble_X, load_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_path = Path(args.model_path or Path(cfg.paths.checkpoints_dir) / "lgbm_fusion_fold0.pkl")
    features_dir = Path(cfg.paths.features_dir) / args.split
    output_path = Path(args.output or Path(cfg.paths.submissions_dir) / "submission_physical_fold0.csv")

    template_df = load_public_test_dataframe(cfg)
    train_df = load_train_dataframe(cfg)
    fallback_score = float(train_df[cfg.data.label_col].mean())

    feature_data = load_features(features_dir)
    X_test = assemble_X(feature_data, cfg.fusion.feature_sources)
    test_ids = feature_data["ids"].astype(str)

    model = LGBMFusion.load(model_path)
    scores = model.predict(X_test)

    scored_df = pd.DataFrame({cfg.data.id_col: test_ids, cfg.data.label_col: scores})
    submission = template_df[[cfg.data.id_col]].copy()
    submission = submission.merge(scored_df, on=cfg.data.id_col, how="left")
    fallback_rows = int(submission[cfg.data.label_col].isna().sum())
    submission[cfg.data.label_col] = submission[cfg.data.label_col].fillna(fallback_score)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"scored_rows={len(scored_df)}")
    print(f"fallback_rows={fallback_rows}")
    print(f"fallback_score={fallback_score:.10f}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()