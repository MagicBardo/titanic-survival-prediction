from pathlib import Path
import shutil
import json
import joblib

from titanic_ml.paths import MODELS, ARCHIVE

def get_next_version():
    existing = list(
        ARCHIVE.glob("titanic_model_v*.pkl")
    )

    return f"v{len(existing)+1:03}"


def save_best_model(pipeline, accuracy: float, model_path: Path, archive_path: Path):

    metric = "accuracy"
    score = accuracy

    archive_path.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_path = model_path.with_suffix(".json")

    if model_path.exists():
        old_version = len(list(archive_path.glob("*.pkl"))) + 1

        shutil.move(
            model_path,
            archive_path / f"titanic_model_v{old_version:03}.pkl"
        )

        if metadata_path.exists():
            shutil.move(
                metadata_path,
                archive_path / f"metadata_v{old_version:03}.json"
            )

    joblib.dump(
        pipeline,
        model_path
    )

    metadata = {
        "version": get_next_version(),
        "metric": metric,
        "score": score
    }

    with open(metadata_path, "w") as f:
        json.dump(
            metadata,
            f,
            indent=4
        )