from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def create_model(model_name:str) -> ClassifierMixin:
    """
    Creating ML model based on given model name

    Parameters
    ----------
    model_name (str):
        one of 'decision_tree', 'random_forest', 'logistic_regression'

    Returns
    -------
    equivalent model with hard coded parameters

    Future changes
    ------------
    model and model parameters decided by configs file
    """

    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )

    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

    elif model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=42
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )