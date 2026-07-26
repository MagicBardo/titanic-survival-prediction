import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from src.preprocessing import engineer_features, create_preprocessing_pipeline
from src.model import create_model

DATA_PATH = "../data/raw/train.csv"
MODEL_PATH = "../models/titanic_model.pkl"

TARGET = "Survived"

def train(model_type:str) -> None:
    """

    Parameters
    ----------
    model_type (str):
        Which model to use when creating the pipeline

    Returns
    -------
    None
    """

    df = pd.read_csv(DATA_PATH)

    df = engineer_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline = Pipeline([
        (
            "preprocessor",
            create_preprocessing_pipeline()
        ),
        (
            "model",
            create_model(model_type)
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"Accuracy: {accuracy:.2%}"
    )

    joblib.dump(
        pipeline,
        MODEL_PATH
    )


if __name__ == "__main__":
    model = input("Which model would you like to use?:\n")
    print("Training...")
    train(model)
    print("Training finished!")