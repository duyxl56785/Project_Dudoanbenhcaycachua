"""Audit the frozen HOG/SVM checkpoint and evaluate Train, Val, and Test.

No estimator is fit here. Test metrics come from the already published prediction CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.evaluate_models import canonical_sample_id, evaluate, load_predictions, validate_same_test_set
from scripts.train_hog_svm import sha256, write_json
from src.hog_svm import HOGConfig, feature_matrix, image_hog, load_mapping, read_splits


def validate_test_rows(rows: list[dict[str, str]], split: list[dict[str, str]], names: tuple[str, ...]) -> None:
    if len(rows) != len(split):
        raise ValueError("Test prediction count differs from fixed split")
    for prediction, source in zip(rows, split):
        if (prediction["path"] != source["path"] or prediction["true_label_index"] != source["label"]
                or prediction["true_label"] != names[int(source["label"])] or prediction["status"] != "ok"):
            raise ValueError(f"Test prediction does not match fixed split: {source['path']}")
        predicted = int(prediction["predicted_label_index"])
        if predicted not in range(4) or prediction["predicted_label"] != names[predicted]:
            raise ValueError(f"Invalid predicted class: {source['path']}")
        if any(not math.isfinite(float(prediction[f"decision_score_{i}"])) for i in range(4)):
            raise ValueError(f"Non-finite decision score: {source['path']}")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion(matrix: list[list[int]], names: tuple[str, ...], target: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 6.2))
    axis.imshow(matrix, cmap="YlGnBu")
    for row in range(4):
        for col in range(4):
            axis.text(col, row, str(matrix[row][col]), ha="center", va="center", fontsize=13,
                      color="white" if matrix[row][col] > 180 else "black")
    axis.set_xticks(range(4), names, rotation=20, ha="right")
    axis.set_yticks(range(4), names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("HOG + Linear SVM | Test confusion matrix (n=1,356)")
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def plot_comparison(rows: list[dict], target: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(rows))
    width = 0.34
    axis.bar(x - width / 2, [r["accuracy"] for r in rows], width, label="Accuracy", color="#176d43")
    axis.bar(x + width / 2, [r["macro_f1"] for r in rows], width, label="Macro-F1", color="#d06a3a")
    axis.set_xticks(x, [r["model"] for r in rows])
    axis.set_ylim(0, 1.07)
    axis.set_ylabel("Test score")
    axis.set_title("Same fixed Test split (n=1,356)")
    axis.legend()
    axis.grid(axis="y", alpha=.2)
    for bars in axis.containers:
        axis.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def timed_smoke(rows: list[dict[str, str]], data_root: Path, config: HOGConfig,
                scaler, classifier, names: tuple[str, ...]) -> list[dict]:
    samples = [next(row for row in rows if row["label"] == str(index)) for index in range(4)]
    output = []
    for row in samples:
        start = time.perf_counter()
        vector = image_hog(data_root / row["path"], config).reshape(1, -1)
        scaled = scaler.transform(vector)
        predicted = int(classifier.predict(scaled)[0])
        elapsed_ms = (time.perf_counter() - start) * 1000
        scores = classifier.decision_function(scaled)[0]
        output.append({"path": row["path"], "true_label": names[int(row["label"])],
                       "predicted_label": names[predicted], "correct": predicted == int(row["label"]),
                       "inference_ms": elapsed_ms,
                       "decision_scores": {name: float(score) for name, score in zip(names, scores)}})
    # PNG is created from a real Test JPG, not synthetic image content.
    with tempfile.TemporaryDirectory() as temp:
        png = Path(temp) / "real_test_leaf.png"
        with Image.open(data_root / samples[0]["path"]) as image:
            image.save(png)
        vector = image_hog(png, config).reshape(1, -1)
        output.append({"format": "PNG conversion of real Test image", "predicted_label":
                       names[int(classifier.predict(scaler.transform(vector))[0])]})
        broken = Path(temp) / "broken.jpg"
        broken.write_bytes(b"invalid image")
        try:
            image_hog(broken, config)
        except ValueError:
            output.append({"format": "corrupt JPG", "rejected": True})
        else:
            raise AssertionError("Corrupt image unexpectedly accepted")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/hog_svm/evaluation_20261112"))
    parser.add_argument("--resnet-predictions", type=Path, help="Optional NV3 Test CSV for per-image alignment")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    artifacts = project / "results/hog_svm"
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite evaluation: {output}")
    data_root = args.data_root.resolve()
    names = load_mapping(project)
    mapping = dict(enumerate(names))
    metadata = json.loads((artifacts / "model_metadata.json").read_text(encoding="utf-8"))
    checkpoint = artifacts / metadata["checkpoint"]
    test_csv = artifacts / "test_predictions.csv"
    if sha256(checkpoint) != metadata["checkpoint_sha256"] or sha256(test_csv) != metadata["test_predictions_sha256"]:
        raise ValueError("Checkpoint or Test CSV SHA-256 mismatch")
    splits = read_splits(project, data_root)
    if tuple(metadata["class_names"]) != names or {k: len(v) for k, v in splits.items()} != metadata["split_counts"]:
        raise ValueError("Mapping or split counts differ from metadata")
    if any(sha256(project / "data/splits" / f"{split}.csv") != metadata["split_csv_sha256"][split] for split in splits):
        raise ValueError("Split CSV differs from training")
    bundle = joblib.load(checkpoint)  # Verified first; joblib must not load untrusted files.
    if bundle["hog_config"] != metadata["hog_config"] or tuple(bundle["class_names"]) != names or bundle["C"] != metadata["C"]:
        raise ValueError("Checkpoint contents differ from metadata")
    scaler, classifier = bundle["scaler"], bundle["classifier"]
    if int(scaler.n_samples_seen_) != len(splits["train"]) or int(classifier.n_iter_) >= int(classifier.max_iter):
        raise ValueError("Scaler was not fit on Train only or SVM hit iteration limit")
    config = HOGConfig(**bundle["hog_config"])

    evaluated = {}
    timings = {}
    for split in ("train", "val"):
        start = time.perf_counter()
        matrix, true = feature_matrix(splits[split], data_root, config)
        extract_s = time.perf_counter() - start
        start = time.perf_counter()
        pred = classifier.predict(scaler.transform(matrix))
        predict_s = time.perf_counter() - start
        records = [{"id": row["path"], "true": int(actual), "pred": int(predicted)}
                   for row, actual, predicted in zip(splits[split], true, pred)]
        evaluated[split] = evaluate(records, mapping)
        timings[split] = {"hog_extract_seconds": extract_s, "scale_and_predict_seconds": predict_s}
        print(f"{split}: {len(records)} images accuracy={evaluated[split]['accuracy']:.6f} macro-F1={evaluated[split]['macro_f1']:.6f}", flush=True)
    if (abs(evaluated["val"]["accuracy"] - metadata["validation"]["val_accuracy"]) > 1e-12
            or abs(evaluated["val"]["macro_f1"] - metadata["validation"]["val_macro_f1"]) > 1e-12):
        raise ValueError("Validation metrics differ from original selection run")

    with test_csv.open(encoding="utf-8-sig", newline="") as handle:
        raw_test = list(csv.DictReader(handle))
    validate_test_rows(raw_test, splits["test"], names)
    test_rows = load_predictions(test_csv, mapping, expected_split="test")
    # NV5's canonicalizer strips "/raw/color/"; the SVM CSV starts at
    # "raw/color/". Add a slash so the same actual leaf gets the same ID.
    for record, raw in zip(test_rows, raw_test):
        record["id"] = canonical_sample_id("/" + raw["path"], is_path=True)
    cnn_rows = load_predictions(project / "results/cnn/cnn_test_predictions.csv", mapping, expected_split="test")
    validate_same_test_set(test_rows, cnn_rows)
    evaluated["test"] = evaluate(test_rows, mapping)
    official = {"CNN": json.loads((project / "results/cnn_metrics_nv2_handoff.json").read_text(encoding="utf-8")),
                "ResNet18": json.loads((project / "results/resnet18_metrics_nv3_handoff.json").read_text(encoding="utf-8"))}
    supports = [sum(1 for row in splits["test"] if row["label"] == str(i)) for i in range(4)]
    for model, metrics in official.items():
        if metrics["split"] != "test" or metrics["n_samples"] != len(test_rows) or [sum(row) for row in metrics["confusion_matrix"]] != supports:
            raise ValueError(f"{model} handoff is not on the same class support")
        if sum(metrics["confusion_matrix"][i][i] for i in range(4)) / len(test_rows) != metrics["accuracy"]:
            raise ValueError(f"{model} handoff metrics inconsistent")

    alignment = {"shared_test_images": len(test_rows), "svm_cnn_ids_and_labels_match": True,
                 "svm_csv_sha256": sha256(test_csv),
                 "cnn_csv_sha256": sha256(project / "results/cnn/cnn_test_predictions.csv")}
    if args.resnet_predictions:
        resnet_path = args.resnet_predictions.resolve()
        resnet_rows = load_predictions(resnet_path, mapping, expected_split="test")
        validate_same_test_set(test_rows, resnet_rows)
        resnet_computed = evaluate(resnet_rows, mapping)
        if (resnet_computed["confusion_matrix"] != official["ResNet18"]["confusion_matrix"]
                or any(abs(resnet_computed[key] - official["ResNet18"][key]) > 1e-12
                       for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"))):
            raise ValueError("NV3 Test CSV metrics differ from official handoff")
        alignment.update({"svm_resnet18_ids_and_labels_match": True,
                          "resnet18_csv_metrics_match_official_handoff": True,
                          "resnet18_csv_sha256": sha256(resnet_path),
                          "resnet18_csv_source": "local integration worktree generated by NV3 eval script; not added to NV4 branch"})

    comparison = []
    for model, data in (("CNN", official["CNN"]), ("ResNet18", official["ResNet18"]), ("HOG + Linear SVM", evaluated["test"])):
        comparison.append({"model": model, "split": "test", "n_samples": len(test_rows),
                           "accuracy": data["accuracy"], "macro_precision": data["macro_precision"],
                           "macro_recall": data["macro_recall"], "macro_f1": data["macro_f1"],
                           "train_seconds": (official["CNN"].get("train_seconds") if model == "CNN" else
                                             metadata["validation"]["train_seconds"] if model == "HOG + Linear SVM" else ""),
                           "source": ("results/cnn_metrics_nv2_handoff.json" if model == "CNN" else
                                      "results/resnet18_metrics_nv3_handoff.json" if model == "ResNet18" else
                                      "results/hog_svm/test_predictions.csv")})

    output.mkdir(parents=True)
    report = {"checkpoint_sha256": metadata["checkpoint_sha256"], "test_csv_sha256": metadata["test_predictions_sha256"],
              "split_csv_sha256": metadata["split_csv_sha256"], "class_names": names,
              "scaler_samples_seen": int(scaler.n_samples_seen_), "svm_iterations": int(classifier.n_iter_),
              "svm_max_iter": int(classifier.max_iter), "training_seconds_selected": metadata["validation"]["train_seconds"],
              "metrics": evaluated, "evaluation_timings": timings, "comparison": comparison,
              "accuracy_gap_resnet_minus_svm": official["ResNet18"]["accuracy"] - evaluated["test"]["accuracy"],
              "macro_f1_gap_resnet_minus_svm": official["ResNet18"]["macro_f1"] - evaluated["test"]["macro_f1"]}
    write_json(output / "evaluation_metrics.json", report)
    write_json(output / "cross_model_alignment.json", alignment)
    write_json(output / "timed_inference_smoke.json", {"samples": timed_smoke(splits["test"], data_root, config, scaler, classifier, names),
                                                       "timing_note": "Single-call wall time includes image read, HOG, scaling and predict; device-dependent"})
    write_csv(output / "split_metrics.csv", [dict(split=split, **{key: value for key, value in data.items() if key not in ("per_class", "confusion_matrix")},
                                                   correct=sum(data["confusion_matrix"][i][i] for i in range(4)),
                                                   incorrect=data["n_samples"] - sum(data["confusion_matrix"][i][i] for i in range(4)))
                                               for split, data in evaluated.items()],
              ["split", "n_samples", "accuracy", "macro_precision", "macro_recall", "macro_f1", "correct", "incorrect"])
    write_csv(output / "per_class_test.csv", [{"class": name, **evaluated["test"]["per_class"][name],
                                                "wrong": evaluated["test"]["per_class"][name]["support"] - evaluated["test"]["confusion_matrix"][index][index]}
                                               for index, name in enumerate(names)],
              ["class", "precision", "recall", "f1", "support", "wrong"])
    write_csv(output / "confusion_matrix_test.csv", [{"true_class": names[i], **{names[j]: value for j, value in enumerate(row)}}
                                                      for i, row in enumerate(evaluated["test"]["confusion_matrix"])],
              ["true_class", *names])
    write_csv(output / "comparison_three_models.csv", comparison,
              ["model", "split", "n_samples", "accuracy", "macro_precision", "macro_recall", "macro_f1", "train_seconds", "source"])
    for is_correct, filename in ((True, "examples_correct.csv"), (False, "examples_wrong.csv")):
        examples = []
        for index, name in enumerate(names):
            chosen = [row for row in raw_test if (int(row["true_label_index"]) == int(row["predicted_label_index"])) == is_correct
                      and int(row["true_label_index"]) == index][:3]
            examples.extend({"path": row["path"], "true_label": name, "predicted_label": row["predicted_label"]} for row in chosen)
        write_csv(output / filename, examples, ["path", "true_label", "predicted_label"])
    plot_confusion(evaluated["test"]["confusion_matrix"], names, output / "confusion_matrix_test.png")
    plot_comparison(comparison, output / "comparison_three_models.png")
    print(f"test: {len(test_rows)} images accuracy={evaluated['test']['accuracy']:.6f} macro-F1={evaluated['test']['macro_f1']:.6f}", flush=True)


if __name__ == "__main__":
    main()
