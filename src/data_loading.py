import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_cicids_binary(
    csv_path: str = "data/raw/cicids2017.csv",
    sample_size: int = 50000,
    random_state: int = 42,
):
    """
    Load CICIDS-like CSV and prepare a simple binary dataset:
    BENIGN -> 0, any other label -> 1 (attack).

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    print(f"[INFO] Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    print("[INFO] Columns in CSV:")
    print(list(df.columns))

    # Detect label column automatically
    label_col = None
    for col in df.columns:
        col_norm = col.strip().lower()
        if col_norm in ["label", "class", "attack_label", "target"]:
            label_col = col
            break

    if label_col is None:
        raise ValueError(
            "Could not find a label column. "
            "Tried names like 'Label', 'class', 'attack_label'. "
            "Check the printed column list above."
        )

    print(f"[INFO] Using label column: {label_col!r}")

    # Make sure label is string so we can compare nicely
    df[label_col] = df[label_col].astype(str)

    # Create binary label: 0 = BENIGN, 1 = ATTACK (case-insensitive)
    df["binary_label"] = (df[label_col].str.upper() != "BENIGN").astype(int)

    # Drop original label column
    df = df.drop(columns=[label_col])

    # Keep only numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "binary_label" in numeric_cols:
        numeric_cols.remove("binary_label")

    # Force numeric + handle weird values
    X = df[numeric_cols].astype(float)

    # Replace inf / -inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Clip extreme values to a safe range (just in case)
    X = X.clip(-1e9, 1e9)

    # Now handle NaN
    nan_before = X.isna().sum().sum()
    print(f"[INFO] NaN or inf values in features before cleaning: {nan_before}")
    if nan_before > 0:
        print("[INFO] Filling NaN values with 0.0")
        X = X.fillna(0.0)

    y = df["binary_label"]

    print(f"[INFO] Total rows before sampling: {len(X)}")
    print(f"[INFO] Positive (attack) ratio: {y.mean():.4f}")

    # Optional downsampling to keep things light
    if sample_size is not None and len(X) > sample_size:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=sample_size,
            stratify=y,
            random_state=random_state,
        )
        print(f"[INFO] After sampling: {len(X)} rows")

    # Train / val / test split: 60 / 20 / 20
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        stratify=y,
        random_state=random_state,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state,
    )

    print(f"[INFO] Train size: {len(X_train)}")
    print(f"[INFO] Val size:   {len(X_val)}")
    print(f"[INFO] Test size:  {len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
