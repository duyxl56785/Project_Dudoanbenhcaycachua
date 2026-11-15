# Nội dung cập nhật báo cáo và slide sau merge

> Dùng tài liệu này để bổ sung báo cáo/slide; không thay thế `results/comparison.csv`
> hay tự làm tròn ngược các số liệu gốc. Tất cả các kết quả chính bên dưới dùng
> split Test cố định gồm **1.356 ảnh**, theo mapping `Healthy`, `Early Blight`,
> `Late Blight`, `Bacterial Spot`.

## 1. Giao thức đánh giá

- Dataset được NV1 chốt với Train/Validation/Test = **4.115 / 1.031 / 1.356**.
  Mỗi mô hình fit tiền xử lý và mô hình trên Train; chọn checkpoint/cấu hình bằng
  **Validation macro-F1**; Test chỉ dùng để báo cáo sau khi lựa chọn đã chốt.
- Accuracy cho biết tỷ lệ dự đoán đúng toàn bộ. Macro Precision, Macro Recall và
  Macro-F1 lấy trung bình đều trên bốn lớp, vì vậy không để lớp có nhiều ảnh hơn
  che khuất chất lượng lớp khó như Early Blight.
- Cùng sample ID, nhãn thật và mapping Test đã được đối chiếu trong
  `scripts/evaluate_models.py`. Không dùng Test để chọn lại CNN, ResNet18 hoặc
  cấu hình HOG+HSV+RBF-SVM chính.

## 2. Bảng so sánh chính

| Mô hình | Chọn theo Validation | Accuracy Test | Macro Precision | Macro Recall | Macro-F1 | Đúng / 1.356 |
|---|---|---:|---:|---:|---:|---:|
| CNN từ đầu | `val_macro_f1=0,976836` | **96,98%** | 96,54% | 96,47% | **96,50%** | 1.315 |
| ResNet18 transfer learning | `val_macro_f1=0,980859` | **98,67%** | 98,53% | 98,27% | **98,40%** | 1.338 |
| HOG + Linear SVM V1 (NV4 baseline) | `val_macro_f1=0,695379` | **73,89%** | 69,56% | 69,72% | **69,40%** | 1.002 |
| HOG + HSV + RBF-SVM (NV5) | `val_macro_f1=0,879552` | **92,85%** | 92,51% | 90,25% | **90,96%** | 1.259 |

Nguồn số liệu: `results/comparison.csv`,
`results/cnn_metrics_nv2_handoff.json`,
`results/resnet18_metrics_nv3_handoff.json`,
`results/hog_svm_metrics_nv4_handoff.json` và
`results/hog_svm_nv5_experiment/metrics.json`.

### Nhận xét có thể trình bày

- ResNet18 có Macro-F1 cao nhất. So với CNN, chênh lệch là **1,70 điểm phần
  trăm Accuracy** và **1,89 điểm phần trăm Macro-F1** trên cùng Test. Transfer
  learning là lời giải thích kỹ thuật hợp lý, nhưng không chứng minh quan hệ
  nhân quả khi chưa lặp nhiều seed hoặc kiểm định ngoài miền.
- HOG + HSV + RBF-SVM cao hơn HOG + Linear SVM V1 **18,95 điểm phần trăm
  Accuracy** và **21,56 điểm phần trăm Macro-F1**. Kết quả cho thấy bổ sung màu
  HSV và kernel phi tuyến có ích cho ảnh lá, nhưng vẫn thấp hơn hai mô hình học
  sâu.
- Với HOG + Linear SVM, `decision_score` không phải xác suất; không diễn giải
  điểm này là độ tin cậy softmax.

## 3. Phân tích lỗi từ confusion matrix

| Mô hình | Số ảnh sai | Nhầm lẫn nổi bật |
|---|---:|---|
| CNN | 41 | Early Blight → Late Blight: 14; Late Blight → Early Blight: 12 |
| ResNet18 | 18 | Early Blight → Late Blight: 6; Early Blight → Healthy: 3 |
| HOG + Linear SVM V1 | 354 | Early Blight → Late Blight: 68; Early Blight → Bacterial Spot: 44 |
| HOG + HSV + RBF-SVM | 97 | Early Blight → Late Blight: 51; Late Blight → Early Blight: 11 |

Early Blight và Late Blight là cặp khó nhất xuyên suốt các mô hình. HOG + HSV +
RBF-SVM cải thiện mạnh số lỗi tổng thể, nhưng F1 Early Blight vẫn chỉ **77,89%**;
đây là lý do cần báo cáo Macro-F1 cùng Accuracy.

## 4. Ghi chú bắt buộc về các thực nghiệm sau khi đã xem Test

- HOG + HSV + PCA + RBF-SVM (Accuracy 92,11%, Macro-F1 90,08%) là thực nghiệm
  **thăm dò**, vì kết quả Test không-PCA 92,85% đã được xem trước. Ba nhật ký
  34 / 38 / 34 cấu hình được tách riêng trong
  `results/traditional_svm_search/RUN_HISTORY.md`; không cộng dồn và không dùng
  PCA để thay thế kết luận chính.
- PR #19 bổ sung **HOG + Linear SVM V2**: Accuracy 79,13%, Macro-F1 73,59%,
  đúng 1.073/1.356. V2 chọn cấu hình theo Validation macro-F1 (0,743165), nhưng
  được phát triển sau khi Test V1 đã được xem. Vì vậy chỉ trình bày V2 ở phụ lục
  hoặc phần phát triển tiếp theo, không gọi là một holdout Test độc lập mới.
  Nguồn: `results/hog_svm_v2/README.md` và `test_metrics.json`.

## 5. Hạn chế và hướng phát triển

- PlantVillage chủ yếu là ảnh trong điều kiện kiểm soát; không suy rộng trực
  tiếp kết quả sang ảnh ngoài ruộng.
- Hệ thống chỉ có bốn lớp; ảnh bệnh/cây ngoài phạm vi vẫn bị ép vào một nhãn.
- Chưa có external test set, nhiều seed, calibration xác suất hoặc đánh giá bệnh
  mới chớm. Cần bộ ảnh thực địa độc lập trước khi tuyên bố khả năng ứng dụng
  ngoài thực tế.
- Nếu tiếp tục tối ưu SVM sau khi đã xem Test, phải dùng một holdout mới hoặc
  external validation; không dùng tiếp Test hiện tại để chọn tham số.

## 6. Minh chứng kỹ thuật

- PR #18: đánh giá HOG+HSV+RBF-SVM, checkpoint đã bàn giao ngoài Git và có
  checksum: <https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/18>.
- PR #19: HOG + Linear SVM V2 của NV4:
  <https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/19>.
- Chi tiết hash/checkpoint NV5: `results/hog_svm_nv5_experiment/CHECKPOINT_HANDOFF.md`.
- Phân tích và confusion matrix đầy đủ: `results/analysis.md`.
