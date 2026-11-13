# Handover plan for the NV5 SVM checkpoint

The current branch history contains `hog_hsv_rbf_svm.joblib` (25,157,911 bytes,
SHA-256 `d40cf8af63242ebfe8e149eba333419de2c2551999589b7df2d5710db21cfcd5`).
It is referenced by `metrics.json` and `execution_log.json`.

## Current status: BLOCKED by required authorization

No external copy has been uploaded and no Git history has been rewritten. Before
merge, the repository owner or group lead must choose and authorize one of these
paths:

1. **GitHub Release:** grant a maintainer permission to create a release and
   upload the checkpoint asset, then record its URL and SHA-256 here. After the
   asset is verified, authorize a history rewrite/force-push of this feature
   branch to remove the binary from its existing commits.
2. **Git LFS:** confirm that Git LFS is enabled and permitted for this repository,
   then authorize a migration and force-push. Merely adding LFS now does not
   remove the already committed binary from history.

Do not merge the branch while the reviewer requires the binary to be removed
from its commit history. The checkpoint itself and all measured metrics must not
be replaced without hash verification.
