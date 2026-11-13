# NV5 traditional-SVM run history

This file separates historical runs. Do not add their trial counts together.

| Order | Evidence | Validation trials | Test status |
|---:|---|---:|---|
| 1 | Commit `959f1b3` (`best_config.json` in that commit) | **34** | PCA(128)+RBF-SVM won by Validation macro-F1 0.883470. Its Test result, 0.921091 Accuracy / 0.900837 macro-F1, was recorded after the non-PCA Test result had already been viewed. |
| 2 | Current `validation_search.csv`, `best_config.json`; updated at `53d7c8e` | **38** | Separate Train/Validation expansion. It selected the same PCA(128) configuration. Test was already known, so it is exploratory; its score is not a new or independent Test result. |
| 3 | `../traditional_svm_search_extended/validation_search.csv` | **34** | Separate Train/Validation rerun created after both Test results were known. It selected the same configuration and did not run Test. |

The initial non-PCA HOG + HSV + RBF-SVM evaluation is separately stored in
`../hog_svm_nv5_experiment/`. It is the primary reported improved SVM Test
result: 1,259/1,356 correct, Accuracy 0.928466, macro-F1 0.909566.

All runs use the fixed project split (4,115 Train / 1,031 Validation / 1,356
Test). This chronology addresses evaluation interpretation only; it does not
alter the recorded metrics or artifacts.
