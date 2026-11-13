# NV5 traditional-model search

This artifact records a validation-only search that reuses the fixed project
manifests and does not modify NV4's HOG + LinearSVC pipeline.

## Search protocol

- Train: 4,115 images; Validation: 1,031 images; Test: 1,356 images.
- Scaler and PCA (when used) are fitted on Train only.
- Configurations are ranked by Validation Macro-F1. The retained log contains
  **38 configurations**: 20 HOG + LinearSVC trials across five feature
  families and four C values; 4 HOG + RBF-SVM trials and 12 HOG + PCA +
  LinearSVC trials around the two best feature families; then 2 HOG + PCA +
  RBF-SVM trials.
- The script itself reads only Train/Validation. However, the separate non-PCA
  experiment had already opened Test before this retained 38-trial search was
  run. Therefore this follow-up search is exploratory and its Test result is
  not an untouched holdout.

## Selected configuration

`rgb64_hsv` HOG (64×64 RGB, 9 orientations, 8×8 cells, 2×2 blocks, L2-Hys)
plus 16-bin HSV histograms and 4×4 spatial HSV mean/std, followed by
`StandardScaler`, PCA(128), and RBF-SVM (`C=10`, `gamma=scale`).

Validation Accuracy is **90.3007%** and Validation Macro-F1 is **88.3470%**.
The complete per-trial log is in `validation_search.csv` and the frozen choice
is in `best_config.json`.

## Previously observed Test result for the Validation-selected PCA model

The PCA(128) model was selected by Validation Macro-F1. Its Test score was
computed after the earlier non-PCA experiment had already exposed Test. Treat
this as an exploratory comparison, not a fresh final holdout:

- 1,249/1,356 correct (Accuracy **92.1091%**)
- Macro Precision **90.7366%**, Macro Recall **89.7638%**, Macro-F1 **90.0837%**
- Early Blight F1 **75.5668%**; Late Blight F1 **91.3100%**

The confusion matrix and all per-image predictions are in `test_metrics.json`
and `test_predictions.csv`. The serialized scaler, PCA and SVM are in
`best_traditional_svm.joblib`.

## Comparison with the non-PCA HSV model

Two traditional-model Test results are recorded because both configurations
were previously evaluated and their Test scores have already been viewed:

| Configuration | PCA | Validation Macro-F1 | Test Accuracy | Test Macro-F1 | Role |
|---|---:|---:|---:|---:|---|
| HOG + HSV + RBF-SVM (`C=10`, `gamma=scale`) | No | 0.879552 | 92.8466% | 90.9566% | Earlier non-PCA experiment |
| HOG + HSV + PCA + RBF-SVM (`C=10`, `gamma=scale`) | 128 | **0.883470** | 92.1091% | 90.0837% | Main searched model, selected by Validation Macro-F1 |

The PCA model is the primary model for the systematic search because it had
the higher Validation Macro-F1. The non-PCA model had the higher previously
observed Test score; that Test result does not change the Validation-based
selection. Since both Test results have already been inspected, report them as
previously observed comparisons, not as a fresh untouched final holdout.

`exploratory_comparison_four_models.csv` aligns this secondary result with the
existing CNN, ResNet18 and NV4 HOG + Linear SVM metrics on the same 1,356-image
Test split. The primary project comparison uses the earlier non-PCA result in
`results/comparison.csv`.

## Later Validation-only rerun

`results/traditional_svm_search_extended/` preserves a separate 34-trial rerun
of the search. It was run after both recorded Test results had already been
viewed, selected the same PCA(128) configuration by Validation Macro-F1, and
does not add a new Test result or change the primary SVM conclusion.

## Reproduction

```powershell
$py = 'C:\Users\PC\AppData\Local\Programs\Python\Python312\python.exe'
& $py -m pip install -r requirements-nv5-traditional-svm.txt
& $py scripts/search_traditional_models_nv5.py --data-root data/raw --out-dir results/traditional_svm_search
& $py scripts/evaluate_best_traditional_nv5.py --data-root data/raw --search-dir results/traditional_svm_search
```
