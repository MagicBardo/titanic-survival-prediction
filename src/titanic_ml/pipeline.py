from sklearn.pipeline import Pipeline

from titanic_ml.preprocessing import create_preprocessing_pipeline
from titanic_ml.model import create_model


def create_pipeline(model_name: str) -> Pipeline:
    return Pipeline([
        (
            "preprocessor",
            create_preprocessing_pipeline()
        ),
        (
            "model",
            create_model(model_name)
        )
    ])