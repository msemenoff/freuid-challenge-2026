"""FREUID competition metrics.

Implements:
- AuDET   : Area under the Detection Error Tradeoff curve
- APCER   : Attack Presentation Classification Error Rate
- BPCER   : Bona-Fide Presentation Classification Error Rate
- FREUID Score = 1 - harmonic_mean(1-AuDET, 1-APCER@target_bpcer)
              lower is better.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_curve


def compute_audet(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the Detection Error Tradeoff (DET) curve.

    DET curve: x=BPCER (false-reject rate for bona-fide), y=APCER (false-accept for attacks).
    We compute it equivalently via the ROC curve (swapping axes / using 1-values).

    AuDET is analogous to AuROC but with both axes as error rates.
    AuDET = 1 - AuROC  (for balanced metrics)
    """
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float("nan")
    auc_roc = roc_auc_score(y_true, scores)
    return float(1.0 - auc_roc)


def compute_apcer_at_bpcer(
    y_true: np.ndarray,
    scores: np.ndarray,
    bpcer_target: float = 0.01,
) -> float:
    """APCER at a fixed BPCER operating point.

    BPCER = FRR for bona-fide (y=0) = fraction of genuine docs classified as fraud.
    APCER = FAR for attacks (y=1)   = fraction of fraudulent docs classified as genuine.

    We sweep thresholds using the ROC curve where:
        FPR = fraction of negatives (y=0) above threshold  -> BPCER
        TPR = fraction of positives (y=1) above threshold  -> 1 - APCER
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1, drop_intermediate=False)
    # fpr here = BPCER (rate of genuine docs scored above threshold)
    # tpr here = 1 - APCER
    # Find the threshold where BPCER <= bpcer_target, pick the point closest from above
    valid = fpr <= bpcer_target + 1e-9
    if not valid.any():
        # All thresholds give BPCER > target; return worst-case APCER=1
        return 1.0
    idx = np.where(valid)[0][-1]  # last (highest threshold) point satisfying BPCER constraint
    apcer = float(1.0 - tpr[idx])
    return apcer


def freuid_score(
    y_true: np.ndarray,
    scores: np.ndarray,
    bpcer_target: float = 0.01,
) -> dict[str, float]:
    """Compute all FREUID metrics and return as dict.

    Returns:
        {
            "audet":        float,  # [0,1] lower is better
            "apcer_bpcer1": float,  # [0,1] lower is better
            "freuid":       float,  # [0,1] lower is better
        }
    """
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)

    audet = compute_audet(y_true, scores)
    apcer = compute_apcer_at_bpcer(y_true, scores, bpcer_target)

    g_audet = 1.0 - audet
    g_apcer = 1.0 - apcer

    denom = g_audet + g_apcer
    if denom < 1e-12:
        score = 1.0
    else:
        score = float(1.0 - 2.0 * g_audet * g_apcer / denom)

    return {"audet": audet, "apcer_bpcer1": apcer, "freuid": score}


def find_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    bpcer_target: float = 0.01,
    n_steps: int = 1000,
) -> tuple[float, float]:
    """Sweep decision thresholds to minimize FREUID Score on a validation set.

    Returns:
        (best_threshold, best_freuid_score)
    """
    thresholds = np.linspace(0.0, 1.0, n_steps)
    best_thresh, best_fs = 0.5, float("inf")
    for t in thresholds:
        preds = (scores >= t).astype(float)
        # For threshold-based eval use the raw scores but clamp near 0/1
        pseudo = np.where(preds == 1, scores, 1.0 - scores)
        fs = freuid_score(y_true, scores, bpcer_target)["freuid"]
        if fs < best_fs:
            best_fs = fs
            best_thresh = float(t)
    # The freuid_score doesn't actually use the threshold for ranking,
    # only for the APCER@BPCER operating point – return the threshold
    # that maps to ~bpcer_target BPCER directly from ROC.
    fpr, tpr, thresholds_roc = roc_curve(y_true, scores, pos_label=1, drop_intermediate=False)
    valid = fpr <= bpcer_target + 1e-9
    if valid.any():
        idx = np.where(valid)[0][-1]
        best_thresh = float(thresholds_roc[idx])
    return best_thresh, freuid_score(y_true, scores, bpcer_target)["freuid"]
