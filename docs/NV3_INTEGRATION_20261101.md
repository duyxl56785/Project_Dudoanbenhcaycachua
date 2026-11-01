# NV3 integration audit — 2026-11-01

Scope: PR #7, NV3-owned files only. No merge, main push, real-model retraining,
replacement checkpoint, or edits to NV1/NV2/NV4/NV5 files.

## GitHub observations

Fetched using the platform HTTPS Git proxy:

| Target | Inspected commit |
| --- | --- |
| main | `9fc3d43666ecb30f0770eaeadb0a6fa9801b8808` |
| PR #7 / feature/nv3-resnet18, before this audit's changes | `e8b0e6fe326de78c11d0993ceb8a55ef4abeb8e0` |
| PR #1 / NV4 | `2406b12a383eb5a388774442fe0b364b7e713831` |
| PR #11 / NV5 | `d4b26970e85a010e7ae33a54bce94528142a8745` |

- PR #7 is open and ready for review. Its GitHub conversation records NV2
  approval at `5ce3384`, then dismissal of that stale review at `e8b0e6f`.
  The page states at least one approving review is required. Requesting a new
  review and the merge decision remain with the team; this audit sends no comments.
- Issue #4, #5 and #6 and PR #11 are open. Current NV4/NV5 evidence supersedes
  older comments saying artifacts have not yet been received.
- `git merge-tree --write-tree e8b0e6f origin/main` succeeds without conflicts.
  This is a local merge simulation, not a GitHub mergeability/branch-policy verdict.
- Neither inspected main nor PR #7 contains `.github/workflows/` definitions.
  The public Checks page displays GitHub Actions onboarding rather than run
  results. GitHub GraphQL returned Forbidden; required checks and CI conclusions
  could not be authoritatively enumerated. CI is **UNVERIFIED**, not PASS.

## Integration and changes

NV3 already supplies `build_model(4)` without downloading pretrained weights,
plain tensor state_dict plus metadata, strict loading, shared NV1 Dataset and
RGB/Resize 256/CenterCrop 224/ImageNet normalization. NV4's current API accepts
that interface. The old blocking interface review findings are addressed.

NV5's current `adapt_classification_report` accepts the archived NV3 JSON;
`path,true_label,predicted_label` are accepted CSV aliases. An adapter is already
implemented on NV5; it is not a remaining missing dependency.

This change fixes a concrete safety gap: evaluation previously overwrote metrics,
predictions and confusion-matrix files. It now refuses existing output artifacts
before loading a model. A deliberate reproduction needs a fresh results directory.
New outputs add explicit split, checkpoint SHA-256, available selection metadata
and NV5 metric fields while retaining legacy fields. Original delivered artifacts
and trained weights are untouched. Tests exercise each protected filename,
metric values and the end-to-end hash/schema/CSV contract.

## Current-instance validation

Environment: Python 3.12.14, torch 2.15.0+cpu, torchvision 0.29.2+cpu.
Commands use `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg` and the existing
repository `.venv/bin/python`. Integration copies are outside the checkout,
assembled with `git archive`; no worktree or merge commit was created.

| Check | Outcome |
| --- | --- |
| NV3-only `python -m pytest tests/test_nv3_resnet18.py -q -p no:cacheprovider` | PASS: 11 passed; SKIP: 4 require shared NV1/NV4 implementations absent on this branch |
| main + original NV3 tree | PASS: 22 passed; SKIP: 1 requires NV4 API |
| main + updated NV3 + current NV4 inference/demo/tests | PASS: 53 passed, 19 subtests passed, 0 skipped |
| New NV3 metrics/CSV through actual PR #11 parser/evaluator | PASS: schema and recomputed metrics match on temporary synthetic contract inputs |
| NV5 versioned NV3 handoff JSON | PASS: schema, 1,356 total support, confusion-matrix counts and per-class arithmetic agree |
| Trained checkpoint download/hash/strict-load in this session | SKIP: Drive CONNECT rejected by proxy with HTTP 403 |
| Real Test CSV byte-level comparison / actual-image parity in this session | SKIP: original ZIP/images unavailable |
| GitHub CI | UNVERIFIED: see access limitations above |

No test failures were observed. The combined suite emits 16 Matplotlib/NumPy
deprecation warnings. Synthetic test checkpoints exist only in temporary test
directories and establish software contracts, not disease-classification accuracy.
The original real experiment was not rerun or retrained.

NV4 documents independent hash verification, strict-load, real-image parity and
full Test reproduction. NV5 documents equal Test IDs/targets and metrics.
Their source evidence is linked in `NV3_HANDOFF_20261030.md`. This session read
those records; it did not reproduce their underlying checkpoint/CSV measurements.

## Remaining actions

1. Obtain a current approving review on PR #7 after these NV3 changes; team lead
   decides when to integrate with main. No textual merge conflict was found.
2. To repeat binary validation here, apply the saved network additions for
   `api.github.com`, `drive.google.com`, `drive.usercontent.google.com` in
   environment settings. These are draft changes, not applied runtime access.
   GitHub API authorization must still be checked separately if access is enabled.
3. Download to a new directory, inspect the ZIP safely, verify the expected
   checkpoint SHA-256 before `torch.load(weights_only=True)` and strict-load;
   validate metadata and all four-class real-image parity with NV4. Preserve
   original artifacts and use a new output directory for any Test reproduction.
4. Confirm CI/branch rules with authenticated access. Do not treat absent workflow
   files or passing local tests as an approval to merge.
