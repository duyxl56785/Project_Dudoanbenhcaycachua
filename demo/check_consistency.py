"""Compare the demo API with NV2/NV3 inference scripts on the same image.

Use only group-approved, trained checkpoints. This verifies integration, not accuracy.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import InferenceError, class_names, predict_bytes


def compare_results(api_result, script_result, names, tolerance=1e-5):
    scores = script_result.get("probabilities", script_result.get("scores"))
    if not isinstance(scores, dict) or set(scores) != set(names):
        raise InferenceError("Script không trả đủ bốn xác suất theo mapping.")
    if script_result.get("label") != api_result["top_label"]:
        raise InferenceError("Top-1 của script và API khác nhau.")
    try:
        deltas = {name: abs(float(scores[name]) - api_result["probabilities"][name]) for name in names}
    except (TypeError, ValueError) as exc:
        raise InferenceError(f"Xác suất script không phải số: {exc}") from exc
    if any(not math.isfinite(delta) for delta in deltas.values()):
        raise InferenceError("Xác suất script không hữu hạn.")
    if any(delta > tolerance for delta in deltas.values()):
        raise InferenceError(f"Xác suất script và API khác quá {tolerance}: {deltas}")
    return max(deltas.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", choices=("cnn", "resnet18", "both"), default="both")
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f"Không tìm thấy ảnh: {args.image}")
    data = args.image.read_bytes()
    names = class_names()
    models = ("cnn", "resnet18") if args.model == "both" else (args.model,)
    for model_name in models:
        api_result = predict_bytes(data, model_name, device="cpu")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        process = subprocess.run(
            [sys.executable, "-m", f"scripts.infer_{model_name}", str(args.image.resolve())],
            cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
        )
        if process.returncode:
            raise InferenceError(f"Script {model_name} lỗi: {process.stderr.strip()}")
        try:
            script_result = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise InferenceError(f"Script {model_name} không xuất JSON hợp lệ: {exc}") from exc
        max_delta = compare_results(api_result, script_result, names)
        print(f"{model_name}: PASS; top-1={api_result['top_label']}; max |delta|={max_delta:.8g}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except InferenceError as exc:
        print(f"CHƯA ĐẠT/CHƯA THỰC HIỆN: {exc}", file=sys.stderr)
        sys.exit(1)
