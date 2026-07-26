from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

DECISION_TREE = "decision_tree"
RANDOM_FOREST = "random_forest"
LOGISTIC_REGRESSION = "logistic_regression"

def create_model(model_name:str) -> ClassifierMixin:
    """
    Creating ML model based on given model name

    Parameters
    ----------
    model_name (str):
        one of 'decision_tree', 'random_forest', 'logistic_regression'

    Returns
    -------
    sklearn classifier instance with predefined parameters

    Future changes
    ------------
    model and model parameters decided by configs file
    """

    if model_name == DECISION_TREE:
        return DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE
        )

    elif model_name == RANDOM_FOREST:
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE
        )

    elif model_name == LOGISTIC_REGRESSION:
        return LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )