import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict


def _detect_label_column(columns) -> str:
    """
    Finds the 'Label' column in CICIDS CSV (case/space insensitive).
    """
    for col in columns:
        if col.strip().lower() == "label":
            return col
    raise ValueError("Could not find a 'Label' column in CSV header.")


def _basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleanup: replace +/-inf with NaN. (NaNs filled later)
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# ---------------- BINARY VERSION ---------------- #

def load_cicids_binary(
    csv_path: str = "data/raw/cicids2017.csv",
    sample_size: int = 50000,
    random_state: int = 42,
):
    """
    Binary dataset:
    BENIGN -> 0, everything else -> 1.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    print(f"[INFO] Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    label_col = _detect_label_column(df.columns)
    print(f"[INFO] Using label column: {repr(label_col)}")

    df = _basic_cleaning(df)

    labels = df[label_col].astype(str).str.upper().str.strip()
    y = (labels != "BENIGN").astype(int)

    # features only
    X = df.drop(columns=[label_col])

    n_bad = np.isinf(X.values).sum() + np.isnan(X.values).sum()
    print(f"[INFO] NaN or inf values in features before cleaning: {n_bad}")
    X = X.fillna(0.0)

    # optional sampling
    if sample_size is not None and sample_size < len(X):
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=sample_size,
            stratify=y,
            random_state=random_state,
        )
        print(f"[INFO] After sampling: {len(X)} rows")
    else:
        print(f"[INFO] Total rows (no sampling): {len(X)}")

    pos_ratio = float((y == 1).mean())
    print(f"[INFO] Positive (attack) ratio: {pos_ratio:.4f}")

    # 60/20/20 split
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


# ---------------- MULTI-CLASS VERSION ---------------- #

def _map_attack_category(raw_label: str) -> str:
    """
    Maps raw CICIDS labels into simplified attack groups.
    """
    s = raw_label.upper().strip()

    if s == "BENIGN":
        return "BENIGN"

    # DoS / DDoS
    if "DOS" in s or "DDOS" in s:
        return "DOS/DDOS"

    # Port scan
    if "PORTSCAN" in s or "PORT SCAN" in s:
        return "PORT SCAN"

    # Brute force
    if "BRUTE" in s:
        return "BRUTE FORCE"

    # Web attacks
    if "WEB" in s:
        return "WEB ATTACK"

    # Botnet
    if "BOT" in s:
        return "BOTNET"

    # Infiltration
    if "INFILTRATION" in s:
        return "INFILTRATION"

    # fallback
    return s


def load_cicids_multiclass(
    csv_path: str = "data/raw/cicids2017.csv",
    sample_size: int = 50000,
    random_state: int = 42,
):
    """
    Multi-class dataset:
    Keeps individual attack types (DoS, Port Scan, etc.)

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names
    """
    print(f"[INFO] Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    label_col = _detect_label_column(df.columns)
    print(f"[INFO] Using label column: {repr(label_col)}")

    df = _basic_cleaning(df)

    raw_labels = df[label_col].astype(str)
    mapped_labels = raw_labels.map(_map_attack_category)

    # features
    X = df.drop(columns=[label_col])

    # encode to ints
    class_names: List[str] = sorted(mapped_labels.unique().tolist())
    class_to_idx: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    y = mapped_labels.map(class_to_idx).astype(int)

    n_bad = np.isinf(X.values).sum() + np.isnan(X.values).sum()
    print(f"[INFO] NaN or inf values in features before cleaning: {n_bad}")
    X = X.fillna(0.0)

    # optional sampling
    if sample_size is not None and sample_size < len(X):
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=sample_size,
            stratify=y,
            random_state=random_state,
        )
        print(f"[INFO] After sampling: {len(X)} rows")
    else:
        print(f"[INFO] Total rows (no sampling): {len(X)}")

    # class distribution
    counts = pd.Series(y).value_counts().sort_index()
    print("[INFO] Class distribution after sampling:")
    for idx, cnt in counts.items():
        ratio = cnt / len(y)
        print(f"  - {class_names[idx]:<15} -> {cnt:6d} samples ({ratio:.4f})")

    # drop classes with <3 samples to allow stratify split
    rare_indices = counts[counts < 3].index.tolist()
    if len(rare_indices) > 0:
        print("[WARN] Dropping very rare classes (<3 samples):")
        for idx in rare_indices:
            print(f"  - {class_names[idx]}: {counts[idx]} samples")

        y = pd.Series(y)
        mask = ~y.isin(rare_indices)
        X = X[mask]
        y = y[mask]

        # recalc mapping after drop
        counts = pd.Series(y).value_counts().sort_index()
        kept_old_indices = counts.index.tolist()
        new_class_names = [class_names[i] for i in kept_old_indices]

        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_old_indices)}
        y = y.map(remap).astype(int)
        class_names = new_class_names

        print("[INFO] Class distribution after dropping:")
        counts_new = pd.Series(y).value_counts().sort_index()
        for new_idx, cnt in counts_new.items():
            ratio = cnt / len(y)
            print(f"  - {class_names[new_idx]:<15} -> {cnt:6d} samples ({ratio:.4f})")

    # final 60/20/20 split
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

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names
