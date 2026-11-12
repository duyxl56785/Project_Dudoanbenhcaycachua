# NV5 HSV-enhanced HOG/RBF-SVM experiment

This is an additional NV5 experiment; it does not overwrite NV4's original
`results/hog_svm` HOG + LinearSVC handoff.

## Selection protocol

- Fixed manifests: `data/splits/train.csv`, `val.csv`, `test.csv` (4,115 / 1,031 / 1,356 images).
- The configuration was selected on Validation only: RGB HOG (64×64, 9 orientations, 8×8 cells, 2×2 blocks, L2-Hys) concatenated with HSV histograms (16 bins/channel) and 4×4 spatial mean/std statistics, followed by `StandardScaler` fitted on Train and an RBF SVM (`C=10`, `gamma=scale`).
- Validation result: Accuracy 0.9039767, macro-F1 0.8795517. The Test labels were not read during selection.
- The final evaluator fits on Train only and opens Test only for the final prediction pass. See `metrics.json` and `test_predictions.csv`.

## Final Test result

The independently evaluated Test result is 1,259/1,356 correct (92.8466% Accuracy), macro Precision 92.5076%, macro Recall 90.2480%, and macro-F1 90.9566%. Per-class F1 and the confusion matrix are recorded in `metrics.json`; every Test row is traceable in `test_predictions.csv`.

## Reproduction

```powershell
$py = 'C:\Users\PC\AppData\Local\Programs\Python\Python312\python.exe'
& $py scripts/evaluate_hog_svm_experiment.py --data-root data/raw --out-dir results/hog_svm_nv5_experiment
```

The raw archive is the official `tomato_6502_images.zip` from the project data
folder (SHA-256 `303c77726cd56def12de4bec9a86517c6c65abd62b1dca8614f717fcfc227304`)
and is intentionally not committed.
