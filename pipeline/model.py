from typing import Optional
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

def _scale_pos_weight(y):
    # ratio of negatives to positives (for imbalance)
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    return neg / pos

def train_model(X_train, y_train, preproc, seed: int = 42):
    """
    Build a Pipeline(preproc → XGBClassifier) and fit.
    Uses scale_pos_weight to handle class imbalance (anomaly = positive).
    TODO (Student C): tune hyperparams, early_stopping_rounds with a valid set, CV grid, threshold tuning.
    """
    # Add a validation logic
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2
        random_state=seed, 
        stratify=train
    )

    # Balance classes

    spw = _scale_pos_weight(y_train)

    # Tune XGBOOST
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        reg_alpha=0.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=spw,  # key for imbalance
        verbosity=0,
        use_label_encoder=False
    )
    pipe = Pipeline([("preproc", preproc), ("model", clf)])

    # Training with Early Stopping
    pipe.fit(
        X_train_split,
        y_train_split, 
        model__early_stopping_round=30, 
        model__eval_set=[(X_val, y_val)], 
        model__verbose=False
    )
    return pipe
