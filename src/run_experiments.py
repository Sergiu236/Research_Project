import time
from typing import List

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .data_loading import load_cicids_multiclass
from .models import build_logistic_regression, build_random_forest


def train_and_evaluate(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    class_names: List[str],
    name: str,
):
    print("\n======================")
    print(f"TRAINING MODEL: {name}")
    print("======================")

    # Train on train + val
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    print("[INFO] Fitting model...")
    model.fit(X_train_full, y_train_full)

    print("[INFO] Evaluating on test set...\n")

    # Measure full test inference time
    start = time.time()
    y_pred = model.predict(X_test)
    elapsed = time.time() - start

    print("Classification report (multi-class):")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4,
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    n_samples = len(X_test)
    throughput = n_samples / elapsed if elapsed > 0 else float("inf")

    print(f"\n[INFO] Inference time: {elapsed:.4f} seconds")
    print(f"[INFO] Throughput: {throughput:,.1f} flows/second")


def main():
    # Load multi-class dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = load_cicids_multiclass()

    # Make sure labels are Series
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
        y_val = pd.Series(y_val)
        y_test = pd.Series(y_test)

    lr = build_logistic_regression()
    rf = build_random_forest()

    train_and_evaluate(
        lr,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        class_names=class_names,
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
        class_names=class_names,
        name="Random Forest",
    )


if __name__ == "__main__":
    main()
