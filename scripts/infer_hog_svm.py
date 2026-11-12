"""Predict a tomato-leaf image with a verified HOG + LinearSVC artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib

from src.hog_svm import HOGConfig, image_hog


def predict_file(image_path: Path, checkpoint: Path, metadata_path: Path) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 does not match metadata")
    # joblib uses pickle: only load the artifact produced by this project and verified above.
    bundle = joblib.load(checkpoint)
    if tuple(bundle["class_names"]) != tuple(metadata["class_names"]):
        raise ValueError("Class mapping differs from metadata")
    if bundle["hog_config"] != metadata["hog_config"]:
        raise ValueError("HOG config differs from metadata")
    config = HOGConfig(**bundle["hog_config"])
    vector = image_hog(image_path, config).reshape(1, -1)
    scaled = bundle["scaler"].transform(vector)
    index = int(bundle["classifier"].predict(scaled)[0])
    scores = bundle["classifier"].decision_function(scaled)[0]
    names = tuple(bundle["class_names"])
    return {"image": str(image_path), "predicted_index": index, "predicted_label": names[index],
            "decision_scores": {name: float(score) for name, score in zip(names, scores)},
            "note": "Decision scores are not calibrated probabilities"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("results/hog_svm/hog_linear_svm.joblib"))
    parser.add_argument("--metadata", type=Path, default=Path("results/hog_svm/model_metadata.json"))
    args = parser.parse_args()
    print(json.dumps(predict_file(args.image, args.checkpoint, args.metadata), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
