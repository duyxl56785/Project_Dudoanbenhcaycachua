# NV5 SVM checkpoint — verified external handoff

## Artifact and independent verification (14/11/2026)

- **File:** `hog_hsv_rbf_svm.joblib` (25,157,911 bytes).
- **Google Drive:** https://drive.google.com/file/d/1dlLKDcyn0NkBFJBeedyByJBPC5AkXDVr/view
- **Team folder:** https://drive.google.com/drive/folders/1q_PbLN2mBE02U0c1-NFUc015wYQ0SVxP
- **SHA-256:** `d40cf8af63242ebfe8e149eba333419de2c2551999589b7df2d5710db21cfcd5`.
- **Training environment:** Python 3.12.10, scikit-learn **1.6.1**, scikit-image 0.25.2, numpy 2.2.6, joblib 1.4.2 (from `execution_log.json`).
- **Verified:** the uploaded Google Drive file was independently downloaded to a Windows machine. Its 25,157,911 bytes and SHA-256 matched the original checkpoint exactly.

The recorded metrics (`metrics.json`), 1,356-row predictions (`test_predictions.csv`) and `execution_log.json` remain in Git. Their hashes and model label mapping were not modified.

## Obtain and validate locally

Download from the Google Drive file URL above into:

`results/hog_svm_nv5_experiment/hog_hsv_rbf_svm.joblib`

Then run on PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 results/hog_svm_nv5_experiment/hog_hsv_rbf_svm.joblib
python -m pip install -r requirements-nv5-traditional-svm.txt
python -m unittest tests.test_traditional_svm_search_nv5 -v
```

Use Python 3.12 and compare the reported hash to the expected SHA-256 above. Without the downloaded binary, the checkpoint-loading test is skipped deliberately; metadata, CSV and numeric checks still run.

## Git integration decision

The group lead chose **squash merge**, without history rewrite, force-push or Git LFS (see PR #18 discussion). Commit `b74056a` removed the 25 MB checkpoint from the **final PR branch tree**, and `.gitignore` now prevents accidental re-addition.

**Important:** the binary remains reachable in older commits of this feature branch; deleting its latest tree entry does not erase those historical objects. A squash merge uses the branch's final tree and **does not import the feature branch's intermediate commits containing the binary into `main`**. This does not purge the binary from existing feature-branch history. Do not use a standard merge commit or rebase merge for this handoff.

After all checks/reviews pass, the maintainer may squash merge PR #18 using an expected HEAD SHA. No merge is implied by the checkpoint upload alone.
