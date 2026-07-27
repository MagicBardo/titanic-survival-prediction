import pandas as pd
import joblib
import json
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

from titanic_ml.preprocessing import engineer_features, create_preprocessing_pipeline
from titanic_ml.model import create_model, RANDOM_STATE
from titanic_ml.paths import RAW_DATA, MODELS, LOGS
from titanic_ml.pipeline import create_pipeline


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

    cm = confusion_matrix(y_test, predictions)

    data = {
        "timestamp": str(dt.datetime.now()),
        "model": model_name,
        "accuracy": accuracy,
        "true_positive": int(cm[0, 0]),
        "false_negative": int(cm[0, 1]),
        "false_positive": int(cm[1, 0]),
        "true_negative": int(cm[1, 1]),
        "features": df.drop(columns=[TARGET]).columns.to_list(),
        "random_state": RANDOM_STATE,
        "train_size": y_train.shape[0],
        "test_size": y_test.shape[0]
    }

    with open(LOGS / "training.json", "w") as f:
        json.dump(data, f, indent=4)


def main(model_name:str) -> None:
    """

    Parameters
    ----------
    model_name (str):
        Which model to use when creating the pipeline

    Returns
    -------
    None
    """

    df = pd.read_csv(RAW_DATA / "train.csv")

    df = engineer_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    pipeline = create_pipeline(model_name)

    pipeline.fit(
        X_train,
        y_train
    )

    log_accuracy(pipeline, model_name, df,  y_train, X_test, y_test)

    joblib.dump(
        pipeline,
        MODELS / "titanic_model.pkl"
    )


if __name__ == "__main__":
    model = input("Which model would you like to use?:\n")
    print("Training...")
    main(model)
    print(f"Training results stored in {LOGS}")
    print("Training finished!")