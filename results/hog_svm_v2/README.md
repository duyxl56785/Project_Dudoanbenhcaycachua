# HOG + Linear SVM V2 — NV4

V2 là vòng phát triển **sau khi kết quả Test của V1 đã được xem**. Do đó đây là một thực nghiệm bổ sung trên cùng bộ dữ liệu, không phải một phép xác nhận bằng tập Test còn nguyên tính độc lập. Trong V2, cấu hình được chốt bằng Validation macro-F1; Test chỉ được tính một lần sau khi chốt và không được dùng để chỉnh tiếp tham số.

## Dữ liệu và quy trình

- Split cố định của NV1: Train **4.115**, Validation **1.031**, Test **1.356** ảnh. `read_splits()` kiểm tra path, group, SHA-256 và nhãn chéo giữa các split. Hash của ba CSV phải trùng metadata V1 trước khi huấn luyện.
- Mapping: `Healthy`, `Early Blight`, `Late Blight`, `Bacterial Spot`, theo thứ tự chỉ số 0–3 trong `configs/class_mapping.json`.
- Pipeline: ảnh RGB hoặc grayscale → resize bằng PIL bilinear → HOG (`L2-Hys`, block 2×2) → `StandardScaler` fit **chỉ Train** → `LinearSVC` fit **chỉ Train**. Validation chỉ dùng để xếp hạng; Test chỉ dùng để đánh giá cấu hình đã chọn. Đây là Linear SVM, không dùng kernel RBF.
- Cố định seed 42, `dual=False`, `max_iter=150`, `tol=0.01`. Đây là giới hạn CPU có ghi trong từng trial. Hai trial có cảnh báo hội tụ vẫn nằm trong CSV/log nhưng **không đủ điều kiện được chọn**. Phiên bản thư viện và thời gian từng bước có trong `train_log.json`.

## Tìm cấu hình

Baseline V1 là HOG grayscale 64×64, 9 hướng, cell 8×8, 1.764 chiều. V2 chạy **40 trial hoàn chỉnh**: bốn cấu hình HOG × năm giá trị `C` (`0.001`, `0.01`, `0.1`, `1`, `10`) × hai giá trị `class_weight` (`none`, `balanced`).

| Cấu hình HOG | Số chiều | Train matrix | Macro-F1 Validation tốt nhất |
| --- | ---: | ---: | ---: |
| RGB 64×64, 9 hướng, cell 8 | 1.764 | 29,04 MB | 0,727930 |
| Grayscale 96×96, 12 hướng, cell 16 | 1.200 | 19,75 MB | 0,727013 |
| **RGB 96×96, 9 hướng, cell 16** | **900** | **14,81 MB** | **0,743165** |
| Grayscale 128×128, 9 hướng, cell 16 | 1.764 | 29,04 MB | 0,722274 |

Trước lưới trên, 5 trial thăm dò grayscale 96×96, 9 hướng, cell 8 (4.356 chiều) đã chạy và **được giữ nguyên** trong `preliminary_gray96_high_dim/`. Lượt `C=0.1` mất 273,59 giây trên CPU và macro-F1 chỉ 0,669863; lưới nhiều chiều này được dừng vì chi phí tài nguyên. Tổng cộng có **45 lượt thử được ghi**, trong đó 40 lượt của lưới hoàn chỉnh dùng để chọn V2. Không trial nào có kết quả thấp bị xóa.

Lượt được chọn là **#22**: RGB 96×96, HOG 9 hướng, cell 16×16, block 2×2, `C=0.001`, `class_weight=balanced`. Validation Accuracy **0,789525**, macro-Precision **0,741735**, macro-Recall **0,746806**, macro-F1 **0,743165**. Lượt này hội tụ sau 4 vòng, không có cảnh báo. Checkpoint được nạp lại và dự đoán trên ảnh Validation giống hệt trước khi lưu.

## So sánh V1 và V2

| Chỉ số | V1 | V2 |
| --- | ---: | ---: |
| HOG features | grayscale 64×64; 1.764 chiều | RGB 96×96; 900 chiều |
| `C` / `class_weight` | `0.01` / none | `0.001` / balanced |
| Validation Accuracy | 0,744908 | **0,789525** |
| Validation macro-F1 | 0,695379 | **0,743165** |
| Test Accuracy | 0,738938 | **0,791298** |
| Test macro-F1 | 0,693961 | **0,735876** |
| Thời gian train cấu hình chọn | 18,123 giây (scaler + model) | 0,964 giây (chỉ model; scaler fit riêng) |
| Suy luận Test, gồm HOG + model | 1,753 ms/ảnh | 6,363 ms/ảnh |

Thời gian train V1/V2 **không cùng phạm vi đo** nên không thể kết luận V2 train nhanh hơn theo tỷ lệ trong bảng. Thời gian suy luận được đo bằng một lượt chạy cục bộ cho toàn bộ Test với cùng cách tính, phụ thuộc cache và CPU; đây không phải benchmark nhiều lần. V1 được đo lại từ checkpoint đóng băng, dự đoán vẫn khớp CSV gốc (`v1_timing_comparison.json`).

V2 đúng **1.073/1.356**, sai **283**; V1 đúng 1.002, sai 354. Có 160 ảnh V1 sai nhưng V2 đúng và 89 ảnh theo chiều ngược lại. V2 cải thiện Validation macro-F1 **0,047786** và Test Accuracy **0,052360**. Các con số Test chỉ mô tả artifact đã chốt, không dùng để tiếp tục tìm cấu hình.

### Confusion Matrix V2

Hàng là nhãn thật, cột là nhãn dự đoán, theo thứ tự mapping ở trên.

| Thật \ Dự đoán | Healthy | Early Blight | Late Blight | Bacterial Spot |
| --- | ---: | ---: | ---: | ---: |
| Healthy | 313 | 4 | 9 | 4 |
| Early Blight | 40 | 67 | 69 | 39 |
| Late Blight | 4 | 26 | 320 | 25 |
| Bacterial Spot | 17 | 25 | 21 | 373 |

Early Blight vẫn khó phân biệt nhất: recall **67/215 = 31,16%**, thấp hơn V1 (71/215 = 33,02%). Lớp này thường bị nhầm với Late Blight (69 ảnh). V2 cải thiện ba lớp còn lại, nhưng không khắc phục được hoàn toàn nhầm lẫn bệnh sớm. Không diễn giải điểm quyết định của `LinearSVC` thành xác suất.

## Artifact và cách chạy

- Checkpoint thật: `hog_linear_svm_v2.joblib`, **49.418 byte**, SHA-256 `756bee68d15699323c14fcbd7dc5a535c316d27c2c3d948edf0989d3d8ebc23c`.
- CSV dự đoán Test: `test_predictions.csv`, đủ 1.356 path duy nhất, nhãn/path trùng split gốc, không có dòng lỗi; SHA-256 `f337c4c6f1a2b7680aafaa03ecb37d09a9bb32cb302e98b96c0fe329161641e9`.
- `model_metadata.json`: cấu hình, mapping, hash, số mẫu, lựa chọn Validation. `train_log.json` và `validation_experiments.csv`: toàn bộ 40 trial. `test_metrics.json`: Accuracy, macro metrics, Confusion Matrix và thời gian. `preliminary_gray96_high_dim/`: năm trial thăm dò đã dừng.
- Không commit dataset ảnh. `--data-root` trỏ đến thư mục chứa `raw/color/...`, thường là `data/raw` theo hướng dẫn dataset NV1.

```powershell
python -m scripts.train_hog_svm_v2 --data-root data/raw --artifact-dir results/hog_svm_v2
python -m scripts.infer_hog_svm 'data/raw/raw/color/Tomato___healthy/<image>.JPG' --checkpoint results/hog_svm_v2/hog_linear_svm_v2.joblib --metadata results/hog_svm_v2/model_metadata.json
python -m pytest -q
python -m compileall src scripts tests
git diff --check
```

Lệnh train từ chối ghi đè artifact đã có; để tái lập phải dùng một `--artifact-dir` **mới**. Test V2 kiểm tra hash checkpoint/CSV, mapping, split, scaler fit 4.115 Train, nạp lại checkpoint, JPG/PNG thật và ảnh lỗi. Không sửa code CNN, ResNet18, Streamlit hoặc phần HOG+HSV+RBF-SVM của NV5.
