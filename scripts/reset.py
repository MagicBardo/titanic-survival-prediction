import shutil

from src.titanic_ml.paths import DATA, RAW_DATA, PROCESSED_DATA, MODELS, LOGS


def reset_data_folder():
    if DATA.exists():
        shutil.rmtree(DATA)

    RAW_DATA.mkdir(parents=True)
    PROCESSED_DATA.mkdir(parents=True)

    print("Data folder reset successfully.")


def reset_models_folder():
    if MODELS.exists():
        shutil.rmtree(MODELS)

    MODELS.mkdir(parents=True)

    print("Models folder reset successfully.")

def reset_logs_folder():
    if LOGS.exists():
        shutil.rmtree(LOGS)

    LOGS.mkdir(parents=True)

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