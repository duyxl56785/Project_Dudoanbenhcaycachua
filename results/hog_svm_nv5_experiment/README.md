# NV5 HSV-enhanced HOG/RBF-SVM experiment

This is an additional NV5 experiment; it does not overwrite NV4's original
`results/hog_svm` HOG + LinearSVC handoff.

## Selection protocol

- Fixed manifests: `data/splits/train.csv`, `val.csv`, `test.csv` (4,115 / 1,031 / 1,356 images).
- The configuration was selected on Validation only: RGB HOG (64×64, 9 orientations, 8×8 cells, 2×2 blocks, L2-Hys) concatenated with HSV histograms (16 bins/channel) and 4×4 spatial mean/std statistics, followed by `StandardScaler` fitted on Train and an RBF SVM (`C=10`, `gamma=scale`).
- Validation result: Accuracy 0.9039767, macro-F1 0.8795517. The Test labels were not read during selection.
- The final evaluator fits on Train only and opens Test only for the final prediction pass. See `metrics.json` and `test_predictions.csv`.

## Final Test result

This was the first recorded NV5 SVM Test evaluation: 1,259/1,356 correct
(92.8466% Accuracy), macro Precision 92.5076%, macro Recall 90.2480%, and
macro-F1 90.9566%. Per-class F1 and the confusion matrix are recorded in
`metrics.json`; every Test row is traceable in `test_predictions.csv`. A later
PCA search occurred after this Test result had been viewed, so that later result
is documented as exploratory rather than as a fresh holdout.

## Reproduction

```powershell
$py = 'C:\Users\PC\AppData\Local\Programs\Python\Python312\python.exe'
& $py -m pip install -r requirements-nv5-traditional-svm.txt
& $py scripts/evaluate_hog_svm_experiment.py --data-root data/raw --out-dir results/hog_svm_nv5_experiment_rerun
```

The evaluator refuses to replace `metrics.json`, `test_predictions.csv` or the
checkpoint in an existing output directory unless `--overwrite` is supplied
explicitly. Use a new output directory for any reproduction run so the retained
audit artifact is preserved.

The raw archive is the official `tomato_6502_images.zip` from the project data
folder (SHA-256 `303c77726cd56def12de4bec9a86517c6c65abd62b1dca8614f717fcfc227304`)
and is intentionally not committed.

## External checkpoint handoff

The original trained checkpoint is stored **outside Git** in the group Drive:

- [hog_hsv_rbf_svm.joblib (Google Drive)](https://drive.google.com/file/d/1dlLKDcyn0NkBFJBeedyByJBPC5AkXDVr/view)
- Size: **25,157,911 bytes**
- SHA-256: `d40cf8af63242ebfe8e149eba333419de2c2551999589b7df2d5710db21cfcd5`
- Serialized with **Python 3.12.10 / scikit-learn 1.6.1** (see `execution_log.json`). Use `requirements-nv5-traditional-svm.txt` for a compatible environment.

The file was uploaded on 2026-11-14 and independently downloaded from Drive onto Windows; the downloaded bytes have the same SHA-256 as the original checkout. To test the serialized model locally, download it into `results/hog_svm_nv5_experiment/hog_hsv_rbf_svm.joblib` and verify:

```powershell
Get-FileHash -Algorithm SHA256 results/hog_svm_nv5_experiment/hog_hsv_rbf_svm.joblib
```

Without the optional checkpoint download, the Test predictions and metrics remain auditable from committed CSV/JSON; the joblib integration test deliberately skips while the artifact is absent. See [CHECKPOINT_HANDOFF.md](CHECKPOINT_HANDOFF.md) for the precise distribution and merge decision.
