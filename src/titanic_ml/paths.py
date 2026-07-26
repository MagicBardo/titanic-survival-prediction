from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA = PROJECT_ROOT / "data"
RAW_DATA = DATA / "raw"
PROCESSED_DATA = DATA / "processed"
TEMPORARY = DATA / "tmp"
MODELS = PROJECT_ROOT / "models"
LOGS = PROJECT_ROOT / "logs"

