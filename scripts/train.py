import pandas as pd
import joblib
import json
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from src.preprocessing import engineer_features, create_preprocessing_pipeline
from src.model import create_model, RANDOM_STATE

DATA_PATH = "../data/raw/train.csv"
MODEL_PATH = "../models/titanic_model.pkl"
LOG_PATH = "../logs/training.json"

TARGET = "Survived"

def log_accuracy(pipeline, model_name, df,y_train, X_test, y_test) -> None:
    """
    Gets accuracy score from the pipeline and saves it in a JSON file after every training

    Parameters
    ----------
    pipeline
    model_name
    df
    y_train
    X_test
    y_test

    Returns
    -------
    None
    """

    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    data = {
        "timestamp": str(dt.datetime.now()),
        "model": model_name,
        "accuracy": accuracy,
        "features": df.drop(columns=[TARGET]).columns.to_list(),
        "random_state": RANDOM_STATE,
        "train_size": y_train.shape[0],
        "test_size": y_test.shape[0]
    }

    with open(LOG_PATH, "a") as f:
        json.dump(data, f)


def train(model_name:str) -> None:
    """

    Parameters
    ----------
    model_name (str):
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
            create_model(model_name)
        )
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    log_accuracy(pipeline, model_name, df,  y_train, X_test, y_test)

    joblib.dump(
        pipeline,
        MODEL_PATH
    )


if __name__ == "__main__":
    model = input("Which model would you like to use?:\n")
    print("Training...")
    train(model)
    print("Training finished!")