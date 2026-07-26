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
            "Pclass": 1,
            "Name": "Smith, Miss. Example",
            "Sex": "female",
            "Age": 25,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "12345",
            "Fare": 80,
            "Cabin": "B57",
            "Embarked": "C"
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