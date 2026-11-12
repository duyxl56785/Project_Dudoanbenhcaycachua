# NV4 — HOG + Linear SVM baseline (12/11/2026)

Đây là mô hình truyền thống **đã huấn luyện thật** để NV5 so sánh với CNN và ResNet18 trên cùng Test split của NV1. Không dùng ImageNet preprocessing, không gọi decision scores là xác suất. Mô hình chưa được nối vào Streamlit vì giao diện hiện tại hiển thị bốn xác suất; `LinearSVC` chưa được hiệu chỉnh xác suất.

## Dữ liệu và quy tắc

- Nguồn: PlantVillage bốn lớp theo `data/README.md`, ảnh tại `data/raw/raw/color/` (ảnh không được commit).
- Dùng nguyên `data/splits/train.csv`, `val.csv`, `test.csv`: lần lượt **4.115 / 1.031 / 1.356** ảnh. Kiểm toán cục bộ: 6.502 ảnh có trong manifest, 0 đường dẫn/SHA trùng giữa split, 0 ảnh thiếu, 0 ảnh không đọc được.
- Mapping từ `configs/class_mapping.json`: `0 Healthy`, `1 Early Blight`, `2 Late Blight`, `3 Bacterial Spot`.
- Split CSV SHA-256: Train `a95af933f7abae4bd13fe28b58192571d8e78e8721092ef0cc555f4797746afc`; Val `105751ffa430ca522a027807c7173612a6966c3b92cc7d60a4151da49005215b`; Test `69a889672fa4e41504e3117173103832d1bdffede69a2034e4f4093128102b34`.
- Scaler và SVM được fit **chỉ trên Train**. Chọn HOG và `C` theo macro-F1 trên Validation. Sau khi chốt checkpoint mới tạo dự đoán Test; không chọn lại theo Test.

## Cấu hình và kết quả thật

- Sáu lần thử: HOG grayscale 64×64, 9 orientations, 2×2 cells/block, L2-Hys; pixels/cell = 8 hoặc 16; `C` = 0.01, 0.1 hoặc 1.0. Chi tiết toàn bộ lần thử trong `validation_experiments.csv` và thời gian trích xuất trong `train_log.json`.
- [Biểu đồ Validation thật](validation_comparison.png) được vẽ trực tiếp từ sáu dòng của `validation_experiments.csv` để NV5 dùng khi trình bày.
- Chọn **8×8 pixels/cell, C = 0.01**, chiều vector HOG **1.764**. Validation Accuracy **0,7449078565**; macro-F1 **0,6953791314**. Thời gian fit scaler + LinearSVC của lượt chọn **18,12 giây**; trích xuất HOG Train **14,7 giây**, Val **2,8 giây** trên máy này. Các thời gian thay đổi theo phần cứng.
- Checkpoint `hog_linear_svm.joblib` **95.628 byte**, SHA-256 `ec61605eb4bfd897b6ee7756e54c665dbdad6314a78ceb300d14b0e974ef974b`. Đã nạp lại và xác minh dự đoán không đổi. Artifact nhỏ nên được lưu trực tiếp trong PR; chỉ nạp file từ nguồn tin cậy và đối chiếu SHA-256.
- `test_predictions.csv`: **1.356 dòng**, 0 ảnh lỗi. Mỗi dòng có ID, đường dẫn, nhãn thật/dự đoán, cấu hình, trạng thái và bốn **decision scores**. NV5 có thể tính Accuracy, Precision, Recall, macro-F1 và confusion matrix từ file này. Các decision scores **không phải xác suất**. Git giữ nguyên byte CSV qua `.gitattributes`, nên SHA-256 trong metadata kiểm chứng được cả sau khi clone trên Windows.
- `inference_smoke.json`: dự đoán lặp lại trên ảnh Test thật của bốn lớp; mẫu Early Blight đầu tiên bị nhầm thành Late Blight và được giữ nguyên. JPG/PNG thật, ảnh hỏng và thiếu checkpoint đã được thử.

## Chạy lại

Từ gốc repository, dùng Python 3.12 và cài `python -m pip install -r requirements-hog-svm.txt`. Chuẩn bị dataset theo `data/README.md`. Nếu ảnh đang ở checkout khác, truyền đường dẫn đến thư mục chứa `raw/color/`:

```powershell
python -m scripts.train_hog_svm --data-root "D:\duong-dan\data\raw" --artifact-dir results/hog_svm_new
python -m scripts.infer_hog_svm "D:\duong-dan\data\raw\raw\color\Tomato___healthy\anh.JPG"
python -m pytest -q
python -m compileall src tests scripts
git diff --check
```

Lệnh train từ chối ghi đè checkpoint cũ. Với artifact dir mới, lệnh suy luận cần `--checkpoint` và `--metadata` trỏ vào hai tệp tương ứng. Máy chạy thực nghiệm: Python 3.12.0, NumPy 2.5.4, scikit-image 0.26.0, scikit-learn 1.9.1, joblib 1.6.0; các phiên bản cũng ghi trong `model_metadata.json`.

## Bàn giao NV5

Đọc `test_predictions.csv` để đánh giá Test theo cùng split 1.356 ảnh, `validation_experiments.csv` để mô tả chọn tham số, `train_log.json` để lấy thời gian và `model_metadata.json` để kiểm tra SHA-256/mapping. Để tái suy luận, dùng checkpoint đã commit cùng lệnh `scripts.infer_hog_svm`. Một ảnh mẫu Early Blight dự đoán sai, nên tránh mô tả baseline là đúng tuyệt đối. Dataset PlantVillage có điều kiện chụp kiểm soát; kết quả không suy rộng thành chẩn đoán thực địa.

Kiểm thử tại checkout này: **94 passed, 1 skipped, 19 subtests passed**; `compileall` và `git diff --check` đạt.
