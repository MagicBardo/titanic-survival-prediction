from pathlib import Path
import zipfile
import shutil
import kaggle

from src.paths import RAW_DATA, TEMPORARY


print("Started downloading dataset...")

# Description from kaggle page of competition
# will be added as txt file into data/raw/ folder
data_description = """Data Dictionary:
----------------
+----------+---------------------------------------------+------------------------------------------+
| Variable | Definition                                  | Key                                      |
+----------+---------------------------------------------+------------------------------------------+
| survival | Survival                                    | 0 = No, 1 = Yes                          |
| pclass   | Ticket class                                | 1 = 1st, 2 = 2nd, 3 = 3rd                |
| sex      | Sex                                         |                                          |
| age      | Age in years                                |                                          |
| sibsp    | # of siblings / spouses aboard the Titanic  |                                          |
| parch    | # of parents / children aboard the Titanic  |                                          |
| ticket   | Ticket number                               |                                          |
| fare     | Passenger fare                              |                                          |
| cabin    | Cabin number                                |                                          |
| embarked | Port of Embarkation                         | C = Cherbourg, Q = Queenstown, S =       |
|          |                                             | Southampton                              |
+----------+---------------------------------------------+------------------------------------------+

Notes:
------
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children traveled only with a nanny, therefore parch=0 for them."""

# Competition
COMPETITION = "titanic"

# Paths
RAW_DATA.mkdir(parents=True, exist_ok=True)

TEMPORARY.mkdir(parents=True, exist_ok=True)

# Download dataset
kaggle.api.competition_download_files(
    COMPETITION,
    path=TEMPORARY
)

# Extract only train.csv
zip_file = TEMPORARY / "titanic.zip"

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extract("train.csv", RAW_DATA)

# Adding data description for train.csv
with open(RAW_DATA / "description.txt", "w") as file:
    file.write(data_description)

# Cleanup
shutil.rmtree(TEMPORARY)

print("Dataset downloaded successfully.")