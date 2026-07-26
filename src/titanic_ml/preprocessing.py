import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from titanic_ml.paths import PROCESSED_DATA


def age_group(age: int) -> str:
    """
    Helper function for age grouping

    Parameters
    ----------
    age (int)

    Returns
    -------
    age corresponding title (str)
    """

    if age < 1:
        return "Baby"
    elif 1 <= age < 3:
        return "Toddler"
    elif 3 <= age < 10:
        return "Child"
    elif 10 <= age < 18:
        return "Teenager"
    elif 18 <= age < 65:
        return "Adult"
    elif 65 <= age < 100:
        return "Senior"
    else:
        return "untitled"


def save_engineered_features(df: pd.DataFrame) -> None:
    """
    Saves engineered features as csv file for looking at used dataset in well readable format

    Parameters
    ----------
    df (pd.DataFrame):
        first processed dataframe

    Returns
    -------
    None
    """

    df.to_csv(PROCESSED_DATA / "engineered.csv", header=True, index=True)

    print(f"Training data was preprocessed successfully and saved in {PROCESSED_DATA}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Titanic-specific feature engineering

    Clean and preprocess data including dropping and creating rows and columns

    Parameters
    ----------
    df (pd.DataFrame):

    Returns
    -------
    df (pd.DataFrame):
    """

    df = df.drop_duplicates()

    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["Embarked"] = df["Embarked"].fillna("C")

    df["Has_Cabin"] = df["Cabin"].notna().astype(int)

    df["Fare"] = df["Fare"].astype(float)

    df = df.drop(
        labels=["PassengerId", "Ticket", "Cabin"],
        axis=1
    )

    df["Sex"] = df["Sex"].replace(
        {
            "male": "M",
            "female": "F",
            "divers": "D"
        }
    )

    column = df.pop("Has_Cabin")
    df.insert(
        loc=6,
        column="Has_Cabin",
        value=column
    )

    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1

    df = df.drop(
        labels=["SibSp", "Parch"],
        axis=1
    )

    df["Title"] = df["Name"].str.extract(
        pat=r",\s*([^.]*)\."
    )

    df["Title"] = df["Title"].replace(
        {"Mr": "Mr",
         "Mrs": "Mrs",
         "Master": "Master",
         "Miss": "Miss",
         "Dr": "Dr",
         "Rev": "Rev",

         "Major": "Other",
         "Mlle": "Other",
         "Col": "Other",
         "Don": "Other",
         "Mme": "Other",
         "Ms": "Other",
         "Lady": "Other",
         "Sir": "Other",
         "Capt": "Other",
         "the Countess": "Other",
         "Jonkheer": "Other"
         }
    )

    df["Age_Group"] = df["Age"].apply(age_group)

    df = df.drop(
        labels="Name",
        axis=1
    )

    save_engineered_features(df)

    return df


def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Scikit-learn preprocessing pipeline

    Creating preprocessing pipeline for whole pipeline approach,
    Fills missing values (already done by engineer_features(), see command),
    One-Hot Encoding for all categorical features

    Returns
    -------
    ColumnTransformer:
        calls the here defined changes (Filling values, One-Hot Encoding) when used as a pipeline

    """

    numerical_features = [
        "Age",
        "Fare",
        "Family_Size"
    ]

    categorical_features = [
        "Sex",
        "Pclass",
        "Age_Group",
        "Title",
        "Has_Cabin",
        "Embarked"
    ]

    # This step was already done in the engineering processing but still here for logic second safety
    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    categorical_pipeline = Pipeline([
        # This step was already done in the engineering processing but still here for logic second safety
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        # Splitting categories into binary columns
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ])

    return ColumnTransformer([
        (
            "num",
            numerical_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ])