# NV5 traditional-model search

This artifact records a validation-only search that reuses the fixed project
manifests and does not modify NV4's HOG + LinearSVC pipeline.

## Search protocol

- Train: 4,115 images; Validation: 1,031 images; Test: 1,356 images.
- Scaler and PCA (when used) are fitted on Train only.
- Configurations are ranked by Validation Macro-F1. Test labels are not opened
  by `search_traditional_models_nv5.py`.
- **34 configurations** were run: HOG + LinearSVC across five HOG feature
  families and four C values; HOG + RBF-SVM around the best feature; HOG + PCA
  + LinearSVC for 64/128/256 components; and HOG + PCA + RBF-SVM.

## Selected configuration

`rgb64_hsv` HOG (64×64 RGB, 9 orientations, 8×8 cells, 2×2 blocks, L2-Hys)
plus 16-bin HSV histograms and 4×4 spatial HSV mean/std, followed by
`StandardScaler`, PCA(128), and RBF-SVM (`C=10`, `gamma=scale`).

Validation Accuracy is **90.3007%** and Validation Macro-F1 is **88.3470%**.
The complete per-trial log is in `validation_search.csv` and the frozen choice
is in `best_config.json`.

## Final Test result

After selection, the model was fit on Train only and evaluated once on Test:

- 1,249/1,356 correct (Accuracy **92.1091%**)
- Macro Precision **90.7366%**, Macro Recall **89.7638%**, Macro-F1 **90.0837%**
- Early Blight F1 **75.5668%**; Late Blight F1 **91.3100%**

The confusion matrix and all per-image predictions are in `test_metrics.json`
and `test_predictions.csv`. The serialized scaler, PCA and SVM are in
`best_traditional_svm.joblib`.

`comparison_four_models.csv` aligns this result with the existing CNN, ResNet18
and NV4 HOG + Linear SVM metrics on the same 1,356-image Test split.

## Reproduction

```powershell
$py = 'C:\Users\PC\AppData\Local\Programs\Python\Python312\python.exe'
& $py scripts/search_traditional_models_nv5.py --data-root data/raw --out-dir results/traditional_svm_search
& $py scripts/evaluate_best_traditional_nv5.py --data-root data/raw --search-dir results/traditional_svm_search
```
