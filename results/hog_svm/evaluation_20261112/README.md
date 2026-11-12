# Đánh giá kỹ thuật NV4 — HOG + Linear SVM (12/11/2026)

Đánh giá này **không huấn luyện lại** hoặc chọn checkpoint mới. Train/Validation được chạy qua checkpoint đã chốt để đo chỉ số còn thiếu; Test được tính trực tiếp từ `../test_predictions.csv` đã phát hành. Script tái lập: `python -m scripts.evaluate_hog_svm --data-root <thu_muc_data/raw> --resnet-predictions <resnet18_test_predictions.csv>`; cần đầu ra mới thì thêm `--output-dir <thu_muc_moi>`. Script từ chối ghi đè thư mục kết quả cũ.

## Xác thực mô hình và dữ liệu

- 4.115 Train, 1.031 Validation, 1.356 Test; bốn lớp theo `configs/class_mapping.json`. Split CSV/SHA, checkpoint SHA-256 `ec61605eb4bfd897b6ee7756e54c665dbdad6314a78ceb300d14b0e974ef974b` và CSV Test SHA-256 `6e65dffb5f91b066dee3d6de5a5e16d65f84f04f008bac1e8778ebaf89b4dedc` đều được kiểm tra trước khi đánh giá.
- HOG: ảnh xám resize 64×64, 9 hướng, cell 8×8, block 2×2, L2-Hys, vector 1.764 chiều. `StandardScaler` fit trên đúng 4.115 ảnh Train; `LinearSVC(C=0.01, dual=False, max_iter=3000, random_state=42)` dùng một ranh giới tuyến tính cho mỗi lớp. Checkpoint có `n_iter_=8` dưới giới hạn 3.000; log lần huấn luyện không ghi nhận cảnh báo hội tụ. Sáu lượt thử và C=0.01/0.1/1.0 nằm trong `../validation_experiments.csv`.
- Lượt chọn mất **18,12 giây** cho scale + fit; trích xuất HOG Train **14,69 giây**, Val **2,81 giây** trong log gốc. Thời gian này đo trên máy NV4 và không dùng để so tốc độ với GPU của CNN/ResNet18.
- `cross_model_alignment.json` xác nhận từng ID và nhãn thật của **1.356 ảnh Test** khớp giữa CSV SVM, CNN và CSV ResNet18 từ bản tích hợp cục bộ. Metrics tính từ CSV ResNet18 khớp handoff NV3. CSV ResNet18 không được chép sang PR NV4; checksum nguồn đã ghi trong JSON.

## Chỉ số thật

| Split | Số ảnh | Accuracy | Precision macro | Recall macro | F1 macro | Đúng | Sai |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 4.115 | 98,32% | 98,46% | 97,55% | 97,96% | 4.046 | 69 |
| Validation | 1.031 | 74,49% | 69,61% | 69,86% | 69,54% | 768 | 263 |
| Test | 1.356 | **73,89%** | **69,56%** | **69,72%** | **69,40%** | **1.002** | **354** |

Test đúng 1.002/1.356 (**73,89%**), sai 354/1.356 (**26,11%**). Train cao hơn Validation/Test rõ rệt; đây là dấu hiệu mô hình khái quát hóa kém trên split cố định. Không dùng Test để thay đổi HOG hoặc `C`.

| Lớp | Precision | Recall | F1 | Support | Sai |
|---|---:|---:|---:|---:|---:|
| Healthy | 82,20% | 88,18% | 85,09% | 330 | 39 |
| Early Blight | 44,10% | **33,02%** | 37,77% | 215 | **144** |
| Late Blight | 73,37% | 77,87% | 75,55% | 375 | 83 |
| Bacterial Spot | 78,56% | 79,82% | 79,18% | 436 | 88 |

Chi tiết chính xác trong `split_metrics.csv`, `per_class_test.csv`, `confusion_matrix_test.csv` và `evaluation_metrics.json`. [Ma trận nhầm lẫn](confusion_matrix_test.png) cho thấy **68** ảnh Early Blight thành Late Blight; cùng lớp còn bị nhầm Bacterial Spot **44** và Healthy **32** ảnh. Mẫu Early Blight sai ở smoke test gốc được giữ nguyên. Danh sách ảnh đúng/sai theo lớp nằm trong `examples_correct.csv` và `examples_wrong.csv`.

## So sánh trên cùng Test

| Mô hình | Accuracy | Precision macro | Recall macro | F1 macro | Thời gian huấn luyện có log |
|---|---:|---:|---:|---:|---:|
| CNN | 96,98% | 96,54% | 96,47% | 96,50% | 1.118,7 giây (GPU NV2) |
| ResNet18 | **98,67%** | **98,53%** | **98,27%** | **98,40%** | Chưa có số so sánh được trong handoff |
| HOG + Linear SVM | 73,89% | 69,56% | 69,72% | 69,40% | 18,12 giây (CPU NV4, lượt chọn) |

Nguồn CNN/ResNet18: `results/cnn_metrics_nv2_handoff.json` và `results/resnet18_metrics_nv3_handoff.json`; không sửa các file này. [Biểu đồ ba mô hình](comparison_three_models.png) vẽ từ số liệu trên. ResNet18 cao hơn SVM **24,78 điểm phần trăm Accuracy** và **29,00 điểm phần trăm macro-F1**; CNN cao hơn SVM lần lượt **23,08** và **27,11** điểm. ResNet18 cao nhất trong ba artifact hiện có. Thời gian huấn luyện khác phần cứng, phương pháp và phạm vi đo nên không dùng để khẳng định mô hình nào nhanh hơn theo cùng điều kiện.

HOG lấy đặc trưng cạnh/hình dạng trên ảnh xám, còn Linear SVM học ranh giới tuyến tính trên vector cố định. CNN/ResNet18 học đặc trưng trực tiếp từ ảnh màu và quan hệ phi tuyến. Đây là **giải thích kỹ thuật khả dĩ** cho chênh lệch; bộ số liệu hiện có không chứng minh một nguyên nhân duy nhất của từng ảnh sai. Baseline HOG/SVM giúp môn Học máy có đối chứng với mô hình học sâu.

## Suy luận và giới hạn

`timed_inference_smoke.json` ghi bốn ảnh Test chưa dùng để fit, gồm nhãn thật, nhãn dự đoán, bốn **decision scores** và thời gian một lượt đọc ảnh + HOG + scale + predict. Ba ảnh đúng; mẫu Early Blight đầu tiên nhầm Late Blight. PNG được chuyển từ JPG thật để kiểm tra định dạng; JPG hỏng bị từ chối. Thời gian đơn lẻ phụ thuộc máy/cache, không phải benchmark giữa các mô hình. Decision scores **không phải xác suất**.

Phần NV4 trực tiếp thực hiện: HOG, LinearSVC, chọn tham số trên Validation, checkpoint, prediction CSV, script suy luận, kiểm thử và đánh giá kỹ thuật này. NV5 giữ trách nhiệm tổng hợp báo cáo cuối. Không chỉnh checkpoint/mã CNN, ResNet18 hoặc Streamlit.
