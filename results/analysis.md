# Phân tích đánh giá NV5

## Phạm vi và nguồn số liệu

- Split được đánh giá: `test`, gồm **1.356 ảnh**. Checkpoint đã được NV2 chọn
  bằng Validation trước khi chạy Test; NV5 không chọn lại hyperparameter.
- Artifact CNN: `results/cnn_metrics_nv2_handoff.json`, sao chép có ghi nguồn từ
  `results/cnn/cnn_metrics.json` của PR #9, source commit NV2
  `b0b6fe34f5e9137dbb61061dad8dc04e6647f779`.
- NV5 đã chạy lại script với prediction CSV thật của NV2 và metrics JSON. Các
  metrics tính lại khớp số NV2 trong sai số dưới `1e-6`; prediction CSV không
  được chép sang branch NV5 vì thuộc artifact bàn giao của NV2.
- ResNet18: `pending`. PR #7 xác nhận chưa có checkpoint fine-tuned, metrics
  Test hoặc prediction CSV thật; không tự tạo số thay thế.

## Kết quả CNN

| Chỉ số | Giá trị |
|---|---:|
| Số ảnh Test | 1.356 |
| Accuracy | 0,969764 (96,98%) |
| Macro Precision | 0,965416 |
| Macro Recall | 0,964701 |
| Macro F1 | 0,965034 |

Confusion matrix được đọc từ artifact NV2 và kiểm tra lại khi chạy
`scripts/evaluate_models.py`:

| Thực tế \ Dự đoán | Healthy | Early Blight | Late Blight | Bacterial Spot |
|---|---:|---:|---:|---:|
| Healthy | 329 | 0 | 1 | 0 |
| Early Blight | 0 | 198 | 14 | 3 |
| Late Blight | 5 | 12 | 358 | 0 |
| Bacterial Spot | 0 | 2 | 4 | 430 |

Tổng đường chéo là 1.315/1.356 ảnh; có **41 lỗi**, tương đương khoảng 3,02%.
Theo F1 từng lớp, Healthy đạt 0,9910 và Bacterial Spot 0,9896; Early Blight
thấp nhất với 0,9274, còn Late Blight đạt 0,9521. Đây là mô tả kết quả của
checkpoint đã bàn giao, không phải một lần chọn model mới trên Test.

## Cặp lớp dễ nhầm

- Early Blight → Late Blight: 14 ảnh.
- Late Blight → Early Blight: 12 ảnh.
- Late Blight → Healthy: 5 ảnh.
- Bacterial Spot → Late Blight: 4 ảnh.
- Early Blight → Bacterial Spot: 3 ảnh.

Nhầm lẫn hai chiều Early Blight/Late Blight chiếm 26/41 lỗi (xấp xỉ 63,4%),
cho thấy đây là cặp khó phân biệt nhất trong artifact Test hiện có. Không có
ảnh ví dụ được đưa vào branch NV5 vì NV2 chưa bàn giao gói ảnh minh chứng.

## Bảng so sánh và dữ liệu còn thiếu

`results/comparison.csv` giữ một dòng CNN đã tính và một dòng ResNet18 ở trạng
thái `pending`. Số tham số, thời gian train/inference và thiết bị chỉ được điền
khi metadata chủ mô hình bàn giao; NV5 không suy diễn các trường đang trống.

## Giới hạn diễn giải

- PlantVillage chủ yếu là ảnh lá trong điều kiện được kiểm soát; kết quả không
  đại diện trực tiếp cho ảnh ngoài thực địa.
- Mô hình hiện chỉ có bốn lớp. Ảnh cây khác hoặc bệnh ngoài mapping vẫn bị ép
  vào một trong bốn nhãn.
- Chưa có tập đánh giá riêng cho bệnh mới chớm, nên không kết luận khả năng
  “phát hiện sớm”.
- Confusion matrix và metrics không chứng minh khả năng triển khai lâm nghiệp;
  cần dữ liệu thực địa, kiểm định độc lập và hiệu chỉnh xác suất nếu dùng thực tế.
