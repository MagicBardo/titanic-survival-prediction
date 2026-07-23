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

# Cleanup
shutil.rmtree(DOWNLOAD_DIR)

print("Dataset downloaded successfully.")