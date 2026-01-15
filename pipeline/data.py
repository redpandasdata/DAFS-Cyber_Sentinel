import pandas as pd
import logging

EXPECTED_CATEGORICAL = ["protocol_type", "service", "flag"]
TARGET_COL = "class"   # values: "normal" / "anomaly"

logger = logging.getLogger(__name__)

def load_data(path: str = "data/train_data.csv") -> pd.DataFrame:
    """
    Load the intrusion dataset and standardize target to {0,1} with column name 'class'.
    1 = anomaly (positive class), 0 = normal.
    TODO (Student A): add dtype hints, NA handling policy, schema validation, and logging.
    """
    df = pd.read_csv(path)

    # basic sanity checks
    missing = [c for c in EXPECTED_CATEGORICAL if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected categorical columns: {missing}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"CSV missing target column '{TARGET_COL}'")

    # Normalize target → binary 0/1
    mapping = {"anomaly": 1, "normal": 0}
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map(mapping)
    # If already numeric, assume 0/1
    if not set(df[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("Target must be 'normal'/'anomaly' or 0/1 after mapping.")
    
    # dtype hints
    for col in EXPECTED_CATEGORICAL:
        df[col] = df[col].astype("category")
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # N/A handling policy
    if df[TARGET_COL].isnull().any():
        logger.error("Missing values detected in target column")
        raise ValueError("Target column contains missing values")

    if df[EXPECTED_CATEGORICAL].isnull().any().any():
        logger.warning("Missing values in categorical features, filling with 'unknown'")
        df[EXPECTED_CATEGORICAL] = df[EXPECTED_CATEGORICAL].fillna("unknown")

    # schema validation (basic)
    for col in EXPECTED_CATEGORICAL:
        if not pd.api.types.is_categorical_dtype(df[col]):
            raise TypeError(f"Column '{col}' is not categorical after conversion.")
    if not pd.api.types.is_integer_dtype(df[TARGET_COL]):
        raise TypeError(f"Target column '{TARGET_COL}' is not integer after conversion.")
    
    # Log missing values
    na_counts = df.isna().sum()
    if na_counts.any():
        logger.warning(
            "Missing values detected:\n%s",
            na_counts[na_counts > 0]
        )
    else:
        logger.info("No missing values detected")

    return df
