"""Evaluation script – computes FREUID metrics on a saved feature split.

Usage:
    python src/evaluate.py --split val
    python src/evaluate.py --split val --model_path checkpoints/lgbm_fusion_fold0.pkl
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path
import numpy as np

from src.utils.config import load_config
from src.utils.metrics import freuid_score, find_best_threshold
from src.models.fusion import LGBMFusion
from src.train_fusion import assemble_X, load_features


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val")
    p.add_argument("--features_dir", default=None)
    p.add_argument("--model_path", default=None)
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    feat_dir = Path(args.features_dir or cfg.paths.features_dir) / args.split
    model_path = Path(args.model_path or
                      Path(cfg.paths.checkpoints_dir) / "lgbm_fusion_fold0.pkl")

    data = load_features(feat_dir)
    X = assemble_X(data, cfg.fusion.feature_sources)
    labels = data["labels"].astype(float)

    # Load model
    fusion = LGBMFusion.load(model_path)
    scores = fusion.predict(X)

    metrics = freuid_score(labels, scores, cfg.evaluation.bpcer_target)
    thresh, _ = find_best_threshold(labels, scores, cfg.evaluation.bpcer_target)

    print(f"\n{'='*50}")
    print(f"AuDET            : {metrics['audet']:.4f}")
    print(f"APCER @ 1% BPCER : {metrics['apcer_bpcer1']:.4f}")
    print(f"FREUID Score     : {metrics['freuid']:.4f}  (lower=better)")
    print(f"Best threshold   : {thresh:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
