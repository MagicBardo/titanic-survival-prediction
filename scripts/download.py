from pathlib import Path
import zipfile
import shutil
import kaggle

# Getting root for finding the top level data folder
def find_project_root():
    current = Path.cwd()

    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    raise FileNotFoundError("Project root not found")

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
|          |                                             | Southampton                               |
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
Some children travelled only with a nanny, therefore parch=0 for them.
"""

# Competition
COMPETITION = "titanic"

# Paths
ROOT_DIR = find_project_root()

DATA_DIR = ROOT_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_DIR = ROOT_DIR / "data" / "tmp"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Download dataset
kaggle.api.competition_download_files(
    COMPETITION,
    path=DOWNLOAD_DIR
)

# Extract only train.csv
zip_file = DOWNLOAD_DIR / "titanic.zip"

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extract("train.csv", DATA_DIR)

# Adding data description for train.csv
with open(DATA_DIR / "description.txt", "w") as file:
    file.write(data_description)

# Cleanup
shutil.rmtree(DOWNLOAD_DIR)

print("Dataset downloaded successfully.")