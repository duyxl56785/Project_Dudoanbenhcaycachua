# Khung báo cáo đề tài nhận dạng bệnh lá cà chua

> Tài liệu khung do NV5 tổng hợp. Số liệu thực nghiệm chỉ lấy từ artifact có
> nguồn; phần ResNet18 được giữ `pending` khi chưa có checkpoint/metrics thật.

## 1. Giới thiệu và mục tiêu

Đề tài xây dựng hệ thống phân loại ảnh lá cà chua thành bốn lớp: Healthy,
Early Blight, Late Blight và Bacterial Spot. Mục tiêu là so sánh CNN tự xây dựng
với ResNet18 dùng Transfer Learning trên cùng một Test split, sau đó tích hợp
mô hình vào demo. Hệ thống hỗ trợ phân loại ảnh, không phải công cụ chẩn đoán
chắc chắn và không tuyên bố phát hiện bệnh mới chớm.

## 2. Dữ liệu và tiền xử lý

Nguồn dữ liệu là PlantVillage theo hướng dẫn NV1. PR #8 ghi nhận 6.502 ảnh,
chia thành Train 4.115, Validation 1.031 và Test 1.356; mapping nhãn dùng
`configs/class_mapping.json`. Train được phép augmentation; Validation/Test dùng
tiền xử lý tất định. Manifest và split chung phải được giữ nguyên, không chia
lại khi đánh giá.

## 3. Phương pháp

### 3.1 CNN

CNN được NV2 huấn luyện từ đầu, không dùng pretrained weights. Checkpoint được
chọn bằng Validation trước khi chạy Test. Artifact Test bàn giao gồm metrics,
confusion matrix và prediction CSV; NV5 chỉ kiểm định/tổng hợp, không huấn luyện
lại.

### 3.2 ResNet18 Transfer Learning

ResNet18 dùng backbone pretrained, gồm giai đoạn feature extraction và/hoặc
fine-tuning theo cấu hình NV3. Tại thời điểm lập báo cáo, PR #7 chưa bàn giao
checkpoint fine-tuned và metrics Test, vì vậy mọi ô kết quả ResNet18 để
`pending`, không điền số ước lượng.

## 4. Quy trình thí nghiệm và kiểm định

1. Đọc mapping và split chung từ artifact NV1.
2. Nhận prediction CSV/metrics JSON sau khi NV2/NV3 đã chốt checkpoint bằng
   Validation.
3. Chạy `scripts/evaluate_models.py` để kiểm tra schema, nhãn, split, ID mẫu,
   ground truth và tính nhất quán metrics.
4. Tính Accuracy, macro Precision/Recall/F1, F1 từng lớp và confusion matrix.
5. Ghi nguồn file, checkpoint, số mẫu và trường metadata còn thiếu; không dùng
   Test để chọn hyperparameter.

Lệnh với artifact CNN hiện có:

```bash
python -m scripts.evaluate_models \
  --cnn-metrics results/cnn_metrics_nv2_handoff.json \
  --output-dir results
```

Khi nhận đủ prediction CSV của cả hai mô hình:

```bash
python -m scripts.evaluate_models \
  --cnn-predictions results/cnn/cnn_test_predictions.csv \
  --resnet18-predictions results/resnet18/resnet18_test_predictions.csv \
  --output-dir results
```

## 5. Kết quả CNN đã kiểm tra

Trên 1.356 ảnh Test, CNN đạt Accuracy `0,969764`, macro Precision `0,965416`,
macro Recall `0,964701` và macro F1 `0,965034`. Có 1.315 dự đoán đúng và 41
dự đoán sai. F1 từng lớp: Healthy `0,990964`, Early Blight `0,927400`, Late
Blight `0,952128`, Bacterial Spot `0,989643`.

Confusion matrix và phân tích cặp lớp dễ nhầm được trình bày trong
`results/analysis.md`; bảng máy đọc được nằm ở `results/comparison.csv`.

## 6. Đánh giá so sánh

Hiện `comparison.csv` có CNN ở trạng thái `computed` và ResNet18 ở trạng thái
`pending`. Chưa kết luận mô hình nào tốt hơn vì so sánh công bằng cần prediction
Test của cả hai mô hình trên đúng cùng tập mẫu. Số tham số và thời gian cũng
chỉ được báo cáo khi có metadata từ chủ mô hình.

## 7. Hạn chế và hướng phát triển

- PlantVillage có điều kiện chụp tương đối kiểm soát, nên cần đánh giá thêm trên
  ảnh thực địa trước khi triển khai.
- Bốn lớp hiện tại không bao phủ mọi bệnh/cây; ảnh ngoài mapping vẫn bị ép nhãn.
- Chưa có tập bệnh mới chớm và chưa được phép kết luận “phát hiện sớm”.
- Cần NV3 bàn giao checkpoint, metrics JSON, prediction CSV và metadata để hoàn
  tất Gate 3; sau đó cập nhật lại bảng/slide từ cùng một nguồn.
- Có thể bổ sung calibration, nhiều seed và đánh giá ngoài miền sau khi nhóm
  thống nhất giao thức.

## 8. Tài liệu truy vết

- Phân công: `docs/PHAN_CONG.md` và Issue #6.
- Kết quả NV2: PR #9 và `results/cnn_metrics_nv2_handoff.json`.
- Dependency ResNet18: PR #7, hiện chưa đủ artifact Test.
- Script và kiểm thử NV5: `scripts/evaluate_models.py` và
  `tests/test_nv5_evaluate_models.py`.
