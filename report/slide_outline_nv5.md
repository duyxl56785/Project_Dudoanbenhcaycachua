# Nội dung slide thuyết trình NV5 (12 slide)

Đây là nội dung nguồn để dàn vào PowerPoint theo mẫu CMC sau khi nhóm xác nhận
font, màu, logo và thời lượng. Các số liệu dưới đây chỉ dùng artifact đã kiểm
chứng; ô `PENDING` không được thay bằng số ước lượng.

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
| Kết quả Test đã có | **PENDING** metrics/checkpoint thật |

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

## Slide 7 — Kết quả CNN trên Test

| Chỉ số | CNN |
|---|---:|
| Số ảnh | 1.356 |
| Accuracy | 0,969764 |
| Macro Precision | 0,965416 |
| Macro Recall | 0,964701 |
| Macro F1 | 0,965034 |

**Nguồn:** `results/cnn_metrics_nv2_handoff.json`, PR #9; đã chạy lại với
prediction CSV thật, sai khác dưới `1e-6`.

**Ghi chú nói:** Đây là kết quả của checkpoint đã chốt bằng Validation.

## Slide 8 — F1 từng lớp và confusion matrix

| Lớp | F1 |
|---|---:|
| Healthy | 0,990964 |
| Early Blight | 0,927400 |
| Late Blight | 0,952128 |
| Bacterial Spot | 0,989643 |

**Hình:** Chèn `results/cnn_confusion_matrix.svg` hoặc PNG khi có matplotlib.

**Ghi chú nói:** Tổng đường chéo 1.315/1.356; 41 lỗi.

## Slide 9 — Phân tích lỗi

**Trên slide:**

- Early Blight → Late Blight: 14.
- Late Blight → Early Blight: 12.
- Hai chiều cộng 26/41 lỗi ≈ 63,4%.
- Các lỗi tiếp theo: Late Blight → Healthy (5), Bacterial Spot → Late Blight (4).

**Ghi chú nói:** Đây là phân tích từ confusion matrix; chưa có ảnh minh họa
đúng/sai do NV2 chưa bàn giao gói ảnh.

## Slide 10 — Bảng so sánh và trạng thái ResNet18

**Trên slide:**

| Model | Accuracy | Macro F1 | Trạng thái |
|---|---:|---:|---|
| CNN | 0,969764 | 0,965034 | computed |
| ResNet18 | — | — | **PENDING** |

**Chỗ chèn sau:** prediction CSV, metrics JSON, checkpoint/hash, số tham số,
thời gian, confusion matrix của NV3.

**Ghi chú nói:** Chưa kết luận mô hình tốt hơn khi thiếu Test artifact ResNet18.

## Slide 11 — Demo, hạn chế và ứng dụng

**Demo:** Chèn ảnh màn hình/lệnh chạy đã được NV4 xác nhận; ghi rõ checkpoint
thực tế dùng trong demo.

**Hạn chế:** PlantVillage có điều kiện kiểm soát; ảnh ngoài bốn lớp bị ép nhãn;
chưa có dữ liệu bệnh mới chớm; chưa đủ cơ sở triển khai ngoài thực địa.

**Ghi chú nói:** Demo minh họa pipeline, không phải bằng chứng thay thế đánh giá
Test hoặc chẩn đoán nông học.

## Slide 12 — Kết luận và bàn giao

**Trên slide:** CNN đã được kiểm định trên 1.356 ảnh; lỗi tập trung ở Early/Late
Blight; ResNet18 chờ NV3; cần cập nhật đồng thời code, bảng, báo cáo, slide và
demo sau khi đủ artifact.

**Checklist nói:** Không dùng Test chọn model; mọi số liệu có nguồn; nhóm trưởng
review trước merge.

**Tài liệu:** `report/bao_cao_nv5_framework.md`, `results/analysis.md`, Issue #6,
PR #7/#9/#11.

## Quy ước hình/bảng và chỗ chèn

- Mỗi hình ghi chú thích, nguồn file và ngày/commit nếu có.
- Không chụp hoặc chèn ảnh dataset mới nếu chưa được NV1/NV2 bàn giao.
- Khi NV3 bàn giao, chạy lại script với hai prediction CSV rồi thay toàn bộ ô
  `PENDING` từ cùng một `comparison.csv`; không sửa tay từng slide.
