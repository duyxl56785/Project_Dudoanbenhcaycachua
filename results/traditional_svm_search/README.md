# NV5 traditional-model search

This artifact records a validation-only search that reuses the fixed project
manifests and does not modify NV4's HOG + LinearSVC pipeline.

## Run history and protocol

- Fixed manifests: Train 4,115 images; Validation 1,031 images; Test 1,356
  images. Scaler and PCA (when used) are fitted on Train only.
- The runs below are **separate**. Their trial counts must not be added or
  reported as one search:

| Run | Evidence | Trial count | Scope and status |
|---|---|---:|---|
| Original frozen search | commit `959f1b3` | **34** | Validation Macro-F1 selected PCA(128)+RBF-SVM. Its Test result was later recorded after the earlier non-PCA Test result had already been viewed. |
| Expanded retained search | `validation_search.csv`, `best_config.json` | **38** | A separate Train/Validation expansion: 20 LinearSVC trials, 4 RBF trials, 12 PCA+LinearSVC trials and 2 PCA+RBF-SVM trials. It is exploratory because Test was already known. |
| Later rerun | `../traditional_svm_search_extended/` | **34** | A separate Validation-only rerun after both Test results were known; it does not produce a new Test claim. |

`search_traditional_models_nv5.py` reads only Train/Validation. The historical
chronology above means none of the post-Test runs may be described as having an
untouched independent Test holdout. See `RUN_HISTORY.md` for commit-level
traceability.

## Selected configuration

`rgb64_hsv` HOG (64×64 RGB, 9 orientations, 8×8 cells, 2×2 blocks, L2-Hys)
plus 16-bin HSV histograms and 4×4 spatial HSV mean/std, followed by
`StandardScaler`, PCA(128), and RBF-SVM (`C=10`, `gamma=scale`).

The original 34-trial selection and the later 38-trial expansion selected the
same configuration: Validation Accuracy **90.3007%**, Validation Macro-F1
**88.3470%**. The retained 38-trial log is in `validation_search.csv`; the
original 34-trial selection is traceable at commit `959f1b3` as documented in
`RUN_HISTORY.md`.

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
| HOG + HSV + PCA + RBF-SVM (`C=10`, `gamma=scale`) | 128 | **0.883470** | 92.1091% | 90.0837% | Original 34-trial Validation winner; post-Test exploratory comparison |

PCA(128) won the original 34-trial Validation ranking. It cannot replace the
non-PCA 92.8466% result as the primary SVM Test report because that later PCA
comparison happened after the non-PCA Test result was known. Both Test results
are therefore previously observed comparisons, not fresh independent holdouts.

`exploratory_comparison_four_models.csv` aligns this secondary result with the
existing CNN, ResNet18 and NV4 HOG + Linear SVM metrics on the same 1,356-image
Test split. The primary project comparison uses the earlier non-PCA result in
`results/comparison.csv`.

## Later Validation-only rerun

`results/traditional_svm_search_extended/` preserves the separate 34-trial
rerun listed above. It selected the same PCA(128) configuration, but does not
add a new Test result or change the primary SVM conclusion.

## Reproduction

```powershell
$py = 'C:\Users\PC\AppData\Local\Programs\Python\Python312\python.exe'
& $py -m pip install -r requirements-nv5-traditional-svm.txt
& $py scripts/search_traditional_models_nv5.py --data-root data/raw --out-dir results/traditional_svm_search
& $py scripts/evaluate_best_traditional_nv5.py --data-root data/raw --search-dir results/traditional_svm_search
```
