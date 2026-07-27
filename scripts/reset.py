import shutil

from titanic_ml.paths import DATA, RAW_DATA, PROCESSED_DATA, MODELS, LOGS


def recreate_folder_with_gitkeep(folder):
    folder.mkdir(parents=True, exist_ok=True)

    gitkeep = folder / ".gitkeep"

    if not gitkeep.exists():
        gitkeep.touch()


def reset_data_folder():
    if DATA.exists():
        shutil.rmtree(DATA)

    recreate_folder_with_gitkeep(RAW_DATA)
    recreate_folder_with_gitkeep(PROCESSED_DATA)

    print("Data folder reset successfully.")


def reset_models_folder():
    if MODELS.exists():
        shutil.rmtree(MODELS)

    recreate_folder_with_gitkeep(MODELS)

    print("Models folder reset successfully.")

def reset_logs_folder():
    if LOGS.exists():
        shutil.rmtree(LOGS)

    recreate_folder_with_gitkeep(LOGS)

    print("Logs folder reset successfully.")


def reset():
    print("Resetting project...")
    if input(f"This would delete all files and subdirectories of '{DATA}'. Continue? (y/n): ").lower() == "y":
        reset_data_folder()
    if input(f"This would delete all files and subdirectories of '{MODELS}'. Continue? (y/n): ").lower() == "y":
        reset_models_folder()
    if input(f"This would delete all files and subdirectories of '{LOGS}'. Continue? (y/n): ").lower() == "y":
        reset_logs_folder()
    print("Project reset finished.")


if __name__ == "__main__":
    reset()