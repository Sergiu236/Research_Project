from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_logistic_regression():
    """
    Very simple Logistic Regression for binary classification.
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    return model


def build_random_forest():
    """
    Simple Random Forest for binary classification.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    return model
