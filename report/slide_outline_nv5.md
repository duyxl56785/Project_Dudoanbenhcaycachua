# Nội dung slide thuyết trình NV5 (12 slide)

Đây là nội dung nguồn để dàn vào PowerPoint theo mẫu CMC sau khi nhóm xác nhận
font, màu, logo và thời lượng. Các số liệu dưới đây chỉ dùng artifact đã kiểm
chứng. ResNet18 đã được NV5 nhận từ ZIP bàn giao PR #7, kiểm hash checkpoint và
tính lại từ prediction CSV; thời gian ResNet18 vẫn để trống vì artifact không có.

## Slide 1 — Tên đề tài

**Trên slide:** Nhận diện bệnh qua ảnh lá cà chua; so sánh CNN và ResNet18;
Nhóm 4 · Học máy và Khai phá dữ liệu · Trường Đại học CMC.

**Hình/bảng:** Logo/tiêu đề theo mẫu trường nếu được cung cấp.

**Ghi chú nói:** Giới thiệu bài toán phân loại bốn lớp và vai trò NV5 là kiểm
định, tổng hợp báo cáo, không huấn luyện thay các thành viên.

## Slide 2 — Mục tiêu và phạm vi

**Trên slide:** Phân loại Healthy, Early Blight, Late Blight, Bacterial Spot;
so sánh CNN học từ đầu với ResNet18 Transfer Learning; đánh giá trên Test chung;
tích hợp demo.

**Giới hạn:** Không phải chẩn đoán chuyên gia; chưa chứng minh phát hiện sớm.

**Ghi chú nói:** Nhấn mạnh Test chỉ dùng sau khi checkpoint được chọn bằng
Validation.

## Slide 3 — Dữ liệu và mapping nhãn

**Trên slide:** PlantVillage; tổng 6.502 ảnh theo PR #8; mapping lấy từ
`configs/class_mapping.json`.

| Split | Số ảnh |
|---|---:|
| Train | 4.115 |
| Validation | 1.031 |
| Test | 1.356 |

**Nguồn:** PR #8, `docs/DATASET_README.md`, `configs/class_mapping.json`.

**Ghi chú nói:** Train có augmentation; Validation/Test tiền xử lý tất định.

## Slide 4 — Pipeline thực nghiệm

**Trên slide:** `manifest → mapping → transform → model → prediction CSV → NV5`

**Kiểm soát:** cùng mapping, cùng sample ID/ground truth, không chia Test lại,
không chọn hyperparameter trên Test.

**Ghi chú nói:** NV5 kiểm tra schema và consistency trước khi tính metric.

## Slide 5 — Hai hướng mô hình

| CNN | ResNet18 |
|---|---|
| Học từ đầu | Backbone pretrained |
| NV2 phụ trách | NV3 phụ trách |
| Chọn bằng Validation | Feature extraction/fine-tuning bằng Validation |
| Accuracy / Macro-F1 Test | 0,969764 / 0,965034 | 0,986726 / 0,983978 |
| Checkpoint chọn bằng Validation | epoch 24 | fine-tuning, epoch 9 |

**Ghi chú nói:** Không điền kiến trúc/hyperparameter chưa có trong artifact chủ
mô hình.

## Slide 6 — Giao thức đánh giá NV5

**Trên slide:** `scripts/evaluate_models.py` kiểm tra mapping, split, schema,
ID mẫu, ground truth, số mẫu metadata và metrics consistency.

**Metrics:** Accuracy; macro Precision/Recall/F1; per-class F1; confusion matrix;
số tham số/thời gian nếu metadata có.

**Lệnh:**

```bash
python -m scripts.evaluate_models --cnn-metrics results/cnn_metrics_nv2_handoff.json --output-dir results
```

## Slide 7 — Kết quả Test hai mô hình

| Chỉ số | CNN | ResNet18 |
|---|---:|---:|
| Số ảnh | 1.356 | 1.356 |
| Accuracy | 0,969764 | 0,986726 |
| Macro Precision | 0,965416 | 0,985348 |
| Macro Recall | 0,964701 | 0,982732 |
| Macro F1 | 0,965034 | 0,983978 |

**Nguồn:** hai JSON handoff và prediction CSV thật của PR #9/ZIP PR #7; NV5
đã đối chiếu metric dưới sai số `1e-6`.

**Ghi chú nói:** Cả hai checkpoint đã chốt bằng Validation, không dùng Test để
chọn lại model.

## Slide 8 — F1 từng lớp và confusion matrix

| Lớp | CNN F1 | ResNet18 F1 |
|---|---:|---:|
| Healthy | 0,990964 | 0,993958 |
| Early Blight | 0,927400 | 0,962264 |
| Late Blight | 0,952128 | 0,985430 |
| Bacterial Spot | 0,989643 | 0,994259 |

**Hình:** Chèn `results/cnn_confusion_matrix.svg` và
`results/resnet18_confusion_matrix.svg`.

**Ghi chú nói:** CNN đúng 1.315/1.356, ResNet18 đúng 1.338/1.356.

## Slide 9 — Phân tích lỗi

**Trên slide:**

- Early Blight → Late Blight: 14.
- Late Blight → Early Blight: 12.
- Hai chiều cộng 26/41 lỗi ≈ 63,4%.
- Các lỗi tiếp theo: Late Blight → Healthy (5), Bacterial Spot → Late Blight (4).

**Ghi chú nói:** Đây là phân tích từ confusion matrix; chưa có ảnh minh họa
đúng/sai do NV2 chưa bàn giao gói ảnh.

## Slide 10 — Bảng so sánh hai mô hình

**Trên slide:**

| Model | Accuracy | Macro F1 | Trạng thái |
|---|---:|---:|---|
| CNN | 0,969764 | 0,965034 | computed |
| ResNet18 | 0,986726 | 0,983978 | computed |

**Metadata:** CNN 389.924 tham số, train 1.118,7 giây; ResNet18 11.178.564 tham
số, 8.395.780 trainable. Thời gian ResNet18 chưa bàn giao, để trống.

**Ghi chú nói:** ResNet18 cao hơn CNN 0,016962 Accuracy và 0,018944 macro-F1.
ImageNet pretraining/fine-tuning là diễn giải hợp lý, không phải kết luận nhân quả.

## Slide 11 — Demo, hạn chế và ứng dụng

**Demo:** NV4 ghi nhận strict-load CPU, `demo.check_consistency --model both`
PASS với `max |delta| = 0`; chèn ảnh/video NV4 đã xác nhận khi đóng gói cuối.

**Hạn chế:** PlantVillage có điều kiện kiểm soát; ảnh ngoài bốn lớp bị ép nhãn;
chưa có dữ liệu bệnh mới chớm; chưa đủ cơ sở triển khai ngoài thực địa.

**Ghi chú nói:** Demo minh họa pipeline, không phải bằng chứng thay thế đánh giá
Test hoặc chẩn đoán nông học.

## Slide 12 — Kết luận và bàn giao

**Trên slide:** ResNet18 đạt macro-F1 0,983978, cao hơn CNN 0,965034 trên 1.356
ảnh Test; lỗi vẫn tập trung Early/Late Blight; kết quả chỉ đại diện PlantVillage.

**Checklist nói:** Không dùng Test chọn model; mọi số liệu có nguồn; nhóm trưởng
review trước merge.

**Tài liệu:** `report/bao_cao_nv5_framework.md`, `results/analysis.md`, Issue #6,
PR #7/#9/#11.

## Quy ước hình/bảng và chỗ chèn

- Mỗi hình ghi chú thích, nguồn file và ngày/commit nếu có.
- Không chụp hoặc chèn ảnh dataset mới nếu chưa được NV1/NV2 bàn giao.
- Khi có artifact mới, chạy lại script với hai prediction CSV rồi cập nhật toàn
  bộ số từ cùng một `comparison.csv`; không sửa tay từng slide.
