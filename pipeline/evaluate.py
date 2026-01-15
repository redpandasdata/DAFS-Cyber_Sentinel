import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, log_loss
)

def evaluate(model, X_test, y_test) -> dict:
    """
    Compute core binary classification metrics.
    IMPORTANT: anomaly = positive class.
    TODO (Student D): add calibration metrics, per-class metrics, threshold search for F1/recall targets.
    """
    ###Predictions
    y_pred = model.predict(X_test)
    if hasattr(model.named_steps["model"], "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback: some estimators only have decision_function
        if hasattr(model.named_steps["model"], "decision_function"):
            scores = model.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            y_proba = y_pred.astype(float)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) == 2 else None,
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "log_loss" : float(log_loss(y_test, y_proba)),
        "positives_test": int(np.sum(y_test == 1)),
        "negatives_test": int(np.sum(y_test == 0)),
    }

    metrics.update({
    "precision_pos": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
    "recall_pos": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
    "precision_neg": float(precision_score(y_test, y_pred, pos_label=0, zero_division=0)),
    "recall_neg": float(recall_score(y_test, y_pred, pos_label=0, zero_division=0)),
    })

    # --- Threshold optimization (F1) ---
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1, best_t = 0.0, 0.5

    for t in thresholds:
        y_hat = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    metrics["best_threshold_f1"] = float(best_t)
    metrics["best_f1"] = float(best_f1)

    return metrics
