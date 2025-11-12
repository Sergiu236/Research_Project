import time

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .data_loading import load_cicids_binary
from .models import build_logistic_regression, build_random_forest


def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, name: str):
    print(f"\n======================")
    print(f"TRAINING MODEL: {name}")
    print(f"======================")

    # Combine train + val for final training
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    print("[INFO] Fitting model...")
    model.fit(X_train_full, y_train_full)

    print("[INFO] Evaluating on test set...")
    start = time.time()
    y_pred = model.predict(X_test)
    elapsed = time.time() - start

    print("\nClassification report (Attack = 1):")
    print(classification_report(y_test, y_pred, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    if elapsed > 0:
        flows_per_sec = len(X_test) / elapsed
    else:
        flows_per_sec = float("inf")

    print(f"\n[INFO] Inference time on test set: {elapsed:.4f} seconds")
    print(f"[INFO] Approx. throughput: {flows_per_sec:,.1f} flows/second")


def main():
    # 1. Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_cicids_binary(
        csv_path="data/raw/cicids2017.csv",
        sample_size=50000,
        random_state=42,
    )

    # 2. Build models
    log_reg = build_logistic_regression()
    rf = build_random_forest()

    # 3. Train + evaluate
    train_and_evaluate(
        log_reg,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        name="Logistic Regression",
    )

    train_and_evaluate(
        rf,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        name="Random Forest",
    )


if __name__ == "__main__":
    main()
