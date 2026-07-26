import pandas as pd

path = "../data/raw/train.csv"
titanic_dataframe = pd.read_csv(path)


def age_group(age: int) -> str:
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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    df["Age"] = df["Age"].fillna(
        df["Age"].median()
    )

    df["Embarked"] = df["Embarked"].fillna(
        "C"
    )

    df["Has_Cabin"] = df["Cabin"].notna().astype(int)

    df = df.drop(
        labels=["PassengerId", "Ticket", "Fare", "Cabin"],
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

    df = pd.get_dummies(
        df,
        columns=["Sex", "Pclass", "Embarked"],
        dtype=int
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

    df = pd.get_dummies(
        df,
        columns=["Age_Group"],
        dtype=int
    )

    df.drop(
        labels="Name",
        axis=1,
        inplace=True
    )

    df = pd.get_dummies(
        df,
        columns=["Title"],
        drop_first=True,
        dtype=int
    )


    return df


if __name__ == "__main__":
    print("Data preparation started")
    preprocess_data(titanic_dataframe)
    print("Data preparation finished")