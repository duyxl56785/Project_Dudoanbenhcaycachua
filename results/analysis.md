# Phân tích đánh giá NV5

- Split: `test`; Test chỉ được dùng sau khi chốt checkpoint bằng Validation.
- Tạo lúc: `2026-10-29T05:14:35.367760+00:00`.
- Python: `3.14.2`.
- Mapping: `Healthy, Early Blight, Late Blight, Bacterial Spot`.

## Nguồn truy vết

- **CNN**: `results\cnn_metrics_nv2_handoff.json`
  - metadata: `results\cnn_metrics_nv2_handoff.json`
- **ResNet18**: chưa bàn giao artifact, trạng thái `pending`.

## Kết quả

| Mô hình | Trạng thái | Số mẫu | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| CNN | computed | 1356 | 0.9698 | 0.9654 | 0.9647 | 0.9650 |
| ResNet18 | pending | — | — | — | — | — |

Macro F1 cao nhất trong các artifact hiện có: **CNN** (0.9650). Đây không phải quyết định chọn lại checkpoint.

## Cặp lớp dễ nhầm

- Early Blight → Late Blight: 14 mẫu
- Late Blight → Early Blight: 12 mẫu
- Late Blight → Healthy: 5 mẫu
- Bacterial Spot → Late Blight: 4 mẫu
- Early Blight → Bacterial Spot: 3 mẫu
- Bacterial Spot → Early Blight: 2 mẫu
- Healthy → Late Blight: 1 mẫu

## Hạn chế và dữ liệu thiếu

- PlantVillage là dữ liệu có kiểm soát; không suy rộng trực tiếp ra ảnh ngoài thực địa.
- Ảnh ngoài bốn lớp vẫn bị ép vào một nhãn; chưa có đánh giá bệnh mới chớm.
- ResNet18 chỉ được ghi nhận khi NV3 bàn giao metrics/predictions thật; không tạo số liệu thay thế.
- Không dùng Test để chọn hyperparameter hoặc checkpoint.
