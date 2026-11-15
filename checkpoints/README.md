# Checkpoints

Lưu trọng số tốt nhất theo **Validation** của từng mô hình.

GitHub chặn file lớn hơn 100 MB. Nếu checkpoint vượt giới hạn, lưu ở Google Drive hoặc GitHub Release và ghi link bên dưới (link phải được cấp quyền cho cả nhóm và giảng viên).

| File                 | Mô hình  | Người phụ trách | Vị trí lưu  | Commit/phiên bản code | Ghi chú     |
|----------------------|----------|-----------------|-------------|-----------------------|-------------|
| `cnn_best.pt`        | CNN      | NV2 - Nguyễn Văn Duy | Trong Git (thư mục này) | NV2 PR #9; run `e1_lr3e-4` | 1.571.893 byte; epoch 24; SHA-256 `d33c95bbe15e82fa83f1fd954136ebfff6e1cbba1a86fe40af473a377390de1f` |
| `resnet18_best.pt`   | ResNet18 | NV3 - Trần Văn Chính | [Gói NV3 trên Google Drive](https://drive.google.com/file/d/153DOn8zLCiIhnydLoPNDcht3X0EFhKuR/view) (ZIP, kèm `resnet18_best.meta.json`) | NV3 PR #7; commit `5ce3384`; run `20261030_080647_2cbb40` | 44.794.251 byte; epoch 9 (fine-tuning); SHA-256 `61d44ba8ac02ee57c44cbd3835aaac2c657c97924735fea743a60df5f5fdf9e2` |

Checkpoint SVM của NV5 (`hog_hsv_rbf_svm.joblib`, 25 MB) cũng nằm ngoài Git: xem [results/hog_svm_nv5_experiment/CHECKPOINT_HANDOFF.md](../results/hog_svm_nv5_experiment/CHECKPOINT_HANDOFF.md).

Sau khi tải ResNet18: giải nén, chép `resnet18_best.pt` và `resnet18_best.meta.json` vào thư mục này, rồi so SHA-256 (`Get-FileHash -Algorithm SHA256 checkpoints/resnet18_best.pt` trên PowerShell) trước khi chạy demo. Link Drive phải được chia sẻ cho cả nhóm và giảng viên.

Mỗi checkpoint phải đi kèm config tương ứng (`configs/cnn.yaml`, `configs/resnet18.yaml`) và dùng đúng `configs/class_mapping.json`.
