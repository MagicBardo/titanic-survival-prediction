import pandas as pd
import joblib
import json

from colorama import Fore, Style
import datetime as dt
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from titanic_ml.model_storage import save_best_model
from titanic_ml.preprocessing import engineer_features
from titanic_ml.paths import RAW_DATA, MODELS, LOGS, ARCHIVE
from titanic_ml.pipeline import create_pipeline


TARGET = "Survived"

def load_data(path:str) -> pd.DataFrame:
    """
    Loads data from csv file
    Stops program if file is not found

    Parameters
    ----------
    path (str):
        path to csv file,
        should use given paths from paths.py

    Returns
    -------
    pd.DataFrame:
        raw dataframe
    """

    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(Fore.RED + f"File at {path} not found! \nCheck 'data/raw' folder. If nothing there, run 'scripts/download.py'" + Style.RESET_ALL)
        sys.exit()



def log_accuracy(pipeline, model_name, df,y_train, X_test, y_test) -> float:
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
    float
    """

    log_path = LOGS

    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(y_test, predictions)

    cm = confusion_matrix(y_test, predictions)

    params = pipeline.named_steps["model"].get_params()

    trainings = []

    data = {
        "id": len(trainings) + 1,
        "timestamp": str(dt.datetime.now()),
        "model": model_name,
        "parameters": params,
        "accuracy": accuracy,
        "confusion_matrix": {
            "true_positive": int(cm[0, 0]),
            "false_negative": int(cm[0, 1]),
            "false_positive": int(cm[1, 0]),
            "true_negative": int(cm[1, 1])
        },
        "features": df.drop(columns=[TARGET]).columns.to_list(),
        "train_size": y_train.shape[0],
        "test_size": y_test.shape[0]
    }

    if (LOGS / "training_history.json").exists():
        with open((LOGS / "training_history.json"), "r") as f:
            trainings = json.load(f)

    trainings.append(data)

    with open((LOGS / "training_history.json"), "w") as f:
        json.dump(trainings, f, indent=4)

    print(f"Training results logged to {LOGS / 'training_history.json'}")

    return accuracy


def save_model(pipeline) -> None:
    """
    Saves the trained model

    Parameters
    ----------
    pipeline

    Returns
    -------
    None
    """

    joblib.dump(
        pipeline,
        MODELS / "titanic_model.pkl"
    )

    print(f"Trained model saved to {MODELS / 'titanic_model.pkl'}")


def get_current_accuracy(metadata_path):
    if not metadata_path.exists():
        return 0

    with open(metadata_path) as f:
        metadata = json.load(f)

    return metadata["score"]


def main(model_name:str) -> None:
    """
    Loads the dataset and trains the model via a pipeline for direct preprocessing
    Saves trained model and its features and parameters

    Parameters
    ----------
    model_name (str):
        Which model to use when creating the pipeline

    Returns
    -------
    None
    """

    df = load_data(RAW_DATA / "train.csv")

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

    save_model(pipeline)

    accuracy = log_accuracy(pipeline, model_name, df,  y_train, X_test, y_test)
    current_best = get_current_accuracy(MODELS / "metadata.json")

    if accuracy > current_best:
        save_best_model(
            pipeline,
            accuracy,
            MODELS / "titanic_model.pkl",
            ARCHIVE
        )


if __name__ == "__main__":
    model = input("Which model would you like to use?:\n")
    print("Training...")
    main(model)
    print("Training finished!")