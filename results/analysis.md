# Phân tích đánh giá NV5

- Split: `test`; Test chỉ được dùng sau khi chốt checkpoint bằng Validation.
- Tạo lúc: `2026-10-30T03:46:08.855225+00:00`.
- Python: `3.14.2`.
- Mapping: `Healthy, Early Blight, Late Blight, Bacterial Spot`.

## Nguồn truy vết

- **CNN**: NV2 PR #9, results/cnn/cnn_metrics.json, checkpoints/cnn_best.meta.json and results/cnn/runs/e1_lr3e-4/run_info.json
  - metadata: `results/cnn_metrics_nv2_handoff.json`
- **ResNet18**: NV3 PR #7 Google Drive handoff ZIP, downloaded and verified by NV5 on 2026-10-30
  - metadata: `results/resnet18_metrics_nv3_handoff.json`

## Kết quả

| Mô hình | Trạng thái | Số mẫu | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| CNN | computed | 1356 | 0.9698 | 0.9654 | 0.9647 | 0.9650 |
| ResNet18 | computed | 1356 | 0.9867 | 0.9853 | 0.9827 | 0.9840 |

Macro F1 cao nhất trong các artifact hiện có: **ResNet18** (0.9840). Đây không phải quyết định chọn lại checkpoint.

## So sánh CNN và ResNet18

- ResNet18 cao hơn CNN **0.0170 Accuracy** và **0.0189 macro-F1** trên cùng 1356 mẫu Test.
- Cả hai checkpoint đã được chủ mô hình chọn bằng Validation trước khi Test. Chênh lệch này mô tả artifact hiện có, không chứng minh quan hệ nhân quả hay thay thế kiểm định nhiều seed/ảnh thực địa.
- ResNet18 dùng ImageNet pretraining rồi fine-tuning; đây là một giải thích kỹ thuật hợp lý cho kết quả cao hơn CNN học từ đầu, nhưng cần thí nghiệm lặp để khẳng định.

## Cặp lớp dễ nhầm

### CNN

- Đúng 1315/1356; sai 41 ảnh.
- Early Blight → Late Blight: 14 mẫu
- Late Blight → Early Blight: 12 mẫu
- Late Blight → Healthy: 5 mẫu
- Bacterial Spot → Late Blight: 4 mẫu
- Early Blight → Bacterial Spot: 3 mẫu

### ResNet18

- Đúng 1338/1356; sai 18 ảnh.
- Early Blight → Late Blight: 6 mẫu
- Early Blight → Healthy: 3 mẫu
- Late Blight → Early Blight: 3 mẫu
- Early Blight → Bacterial Spot: 2 mẫu
- Bacterial Spot → Early Blight: 2 mẫu

## Hạn chế

- PlantVillage là dữ liệu có kiểm soát; không suy rộng trực tiếp ra ảnh ngoài thực địa.
- Ảnh ngoài bốn lớp vẫn bị ép vào một nhãn; chưa có đánh giá bệnh mới chớm.
- Chưa có thí nghiệm nhiều seed hoặc tập kiểm định ngoài miền để đo độ ổn định chênh lệch CNN–ResNet18.
- Không dùng Test để chọn hyperparameter hoặc checkpoint.
