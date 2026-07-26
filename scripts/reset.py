from pathlib import Path
import shutil


def find_project_root() -> Path:
    current = Path.cwd()

    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    raise FileNotFoundError("Could not find project root")


def reset_data_folder():
    project_root = find_project_root()
    data_dir = project_root / "data"

    # Delete existing data folder
    if data_dir.exists():
        shutil.rmtree(data_dir)

    # Recreate structure
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "processed").mkdir(parents=True)

    print("Data folder reset successfully.")


def reset_models_folder():
    project_root = find_project_root()
    models_dir = project_root / "models"

    if models_dir.exists():
        shutil.rmtree(models_dir)

    print("Models folder reset successfully.")


def reset():
    print("Resetting project...")
    reset_data_folder()
    reset_models_folder()


if __name__ == "__main__":
    reset()