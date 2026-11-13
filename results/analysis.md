# Phân tích đánh giá NV5

- Split: `test`, 1.356 ảnh; mapping: `Healthy, Early Blight, Late Blight, Bacterial Spot`.
- CNN, ResNet18, HOG + Linear SVM và HOG + HSV + RBF-SVM được chọn theo
  Validation trước lần đánh giá Test được ghi nhận. Một tìm kiếm PCA của NV5
  diễn ra sau khi đã xem kết quả Test HOG + HSV + RBF-SVM; kết quả PCA chỉ là
  so sánh thăm dò, không được đưa vào bảng chính hay dùng để chọn mô hình.
- Tạo lúc: `2026-11-12T09:55:48.940640+00:00`.
- Python: `3.12.14`.

## Nguồn truy vết

- **CNN**: NV2 PR #9, results/cnn/cnn_metrics.json, checkpoints/cnn_best.meta.json and results/cnn/runs/e1_lr3e-4/run_info.json
  - metadata: `results/cnn_metrics_nv2_handoff.json`
- **ResNet18**: NV3 PR #7 Google Drive handoff ZIP, downloaded and verified by NV5 on 2026-10-30
  - metadata: `results/resnet18_metrics_nv3_handoff.json`
- **HOG + Linear SVM**: NV4 PR #16, feature/nv4-hog-linear-svm
  - metadata: `results/hog_svm_metrics_nv4_handoff.json`
- **HOG + HSV + RBF-SVM**: NV5 PR #18, lần đánh giá SVM Test đầu tiên được ghi nhận
  - metadata: `results/hog_svm_nv5_experiment/metrics.json`, `execution_log.json`

## Kết quả

| Mô hình | Trạng thái | Số mẫu | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| CNN | computed | 1356 | 0.9698 | 0.9654 | 0.9647 | 0.9650 |
| ResNet18 | computed | 1356 | 0.9867 | 0.9853 | 0.9827 | 0.9840 |
| HOG + Linear SVM | computed | 1356 | 0.7389 | 0.6956 | 0.6972 | 0.6940 |
| HOG + HSV + RBF-SVM | computed | 1356 | 0.9285 | 0.9251 | 0.9025 | 0.9096 |

Macro F1 cao nhất trong các artifact hiện có: **ResNet18** (0.9840). Đây không phải quyết định chọn lại checkpoint.

## So sánh CNN và ResNet18

- ResNet18 cao hơn CNN **0.0170 Accuracy** và **0.0189 macro-F1** trên cùng 1356 mẫu Test.
- Cả hai checkpoint đã được chủ mô hình chọn bằng Validation trước khi Test. Chênh lệch này mô tả artifact hiện có, không chứng minh quan hệ nhân quả hay thay thế kiểm định nhiều seed/ảnh thực địa.
- ResNet18 dùng ImageNet pretraining rồi fine-tuning; đây là một giải thích kỹ thuật hợp lý cho kết quả cao hơn CNN học từ đầu, nhưng cần thí nghiệm lặp để khẳng định.

## HOG + Linear SVM

- HOG + Linear SVM đạt Accuracy **0.7389** và macro-F1 **0.6940** trên 1356 ảnh Test.
- Đây là baseline đặc trưng thủ công: HOG trên ảnh xám kết hợp LinearSVC. Decision score chỉ dùng để chọn nhãn, không phải xác suất.
- Cấu hình và C phải được chọn bằng Validation; Test chỉ dùng một lần sau khi chốt mô hình.

## HOG + HSV + RBF-SVM (NV5)

- Mô hình dùng HOG RGB (64×64) kết hợp histogram và thống kê không gian HSV,
  `StandardScaler` fit trên Train, sau đó là RBF-SVM (`C=10`, `gamma=scale`).
- Cấu hình này có Validation Accuracy **0.9040**, macro-F1 **0.8796**, rồi đạt
  **1.259/1.356** dự đoán đúng (Accuracy **0.9285**, macro-F1 **0.9096**) khi
  Test được mở cho lần đánh giá đầu tiên của chuỗi SVM NV5.
- Nó vượt baseline HOG + Linear SVM, nhưng vẫn thấp hơn CNN và ResNet18. Kết quả
  PCA 0.9211/0.9008 đã được giữ trong thư mục `traditional_svm_search` chỉ để
  truy vết thăm dò, không dùng làm kết quả SVM chính.

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

### HOG + Linear SVM

- Đúng 1002/1356; sai 354 ảnh.
- Early Blight → Late Blight: 68 mẫu
- Early Blight → Bacterial Spot: 44 mẫu
- Late Blight → Early Blight: 39 mẫu
- Late Blight → Bacterial Spot: 35 mẫu
- Bacterial Spot → Early Blight: 35 mẫu

### HOG + HSV + RBF-SVM

- Đúng 1.259/1.356; sai 97 ảnh.
- Early Blight → Late Blight: 51 mẫu.
- Late Blight → Early Blight: 11 mẫu.
- Early Blight → Healthy: 10 mẫu; Early Blight → Bacterial Spot: 6 mẫu.
- Bacterial Spot → Late Blight: 5 mẫu.

## Hạn chế

- PlantVillage là dữ liệu có kiểm soát; không suy rộng trực tiếp ra ảnh ngoài thực địa.
- Ảnh ngoài bốn lớp vẫn bị ép vào một nhãn; chưa có đánh giá bệnh mới chớm.
- Chưa có thí nghiệm nhiều seed hoặc tập kiểm định ngoài miền để đo độ ổn định chênh lệch CNN–ResNet18.
- Không dùng Test để chọn hyperparameter hoặc checkpoint.
- Vì kết quả Test HOG + HSV + RBF-SVM đã được xem trước tìm kiếm PCA, không thể
  coi kết quả PCA sau đó là ước lượng Test hoàn toàn độc lập; cần một test holdout
  mới hoặc kiểm định ngoài miền nếu tiếp tục tối ưu.
