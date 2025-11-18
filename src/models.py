from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_logistic_regression():
    """
    Logistic Regression for both binary and multi-class tasks.
    """
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",   # works well with multinomial
        n_jobs=-1,
        random_state=42,
    )


def build_random_forest():
    """
    Random Forest for binary and multi-class.
    """
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
