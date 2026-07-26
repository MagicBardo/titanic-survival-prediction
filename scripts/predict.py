import joblib
import pandas as pd

from titanic_ml.preprocessing import engineer_features
from titanic_ml.paths import MODELS


def main():

    pipeline = joblib.load(
        MODELS / "titanic_model.pkl"
    )

    passenger = pd.DataFrame([
        {
            "PassengerId": 1,
            "Survived": 0,
            "Pclass": 2,
            "Name": "Smith, Mrs. Example",
            "Sex": "male",
            "Age": 10,
            "SibSp": 7,
            "Parch": 0,
            "Ticket": "12345",
            "Fare": 0,
            "Cabin": None,
            "Embarked": "S"
        }
    ])

    passenger = engineer_features(passenger)

    passenger = passenger.drop(
        columns=["Survived"],
        errors="ignore"
    )

    prediction = pipeline.predict(passenger)

    print(f"Example Passenger {"Survived" if prediction[0] == 1 else "Did not survive"}")


if __name__ == "__main__":
    print("Predicting...")
    main()