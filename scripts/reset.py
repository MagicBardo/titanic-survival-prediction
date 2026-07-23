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


if __name__ == "__main__":
    reset_data_folder()