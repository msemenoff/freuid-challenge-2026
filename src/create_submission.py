"""Create a valid first submission CSV for the released public test set."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.data import load_public_test_dataframe, load_train_dataframe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to submissions/submission_baseline_prior.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    train_df = load_train_dataframe(cfg)
    test_df = load_public_test_dataframe(cfg)

    fraud_rate = float(train_df[cfg.data.label_col].mean())
    submission = pd.DataFrame(
        {
            cfg.data.id_col: test_df[cfg.data.id_col].astype(str),
            cfg.data.label_col: fraud_rate,
        }
    )

    output_path = Path(args.output or Path(cfg.paths.submissions_dir) / "submission_baseline_prior.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    print(f"fraud_rate={fraud_rate:.10f}")
    print(f"rows={len(submission)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()