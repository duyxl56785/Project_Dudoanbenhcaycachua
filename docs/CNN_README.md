# CNN tự xây dựng

Phụ trách: NV2 - Nguyễn Văn Duy. Trạng thái: Đang làm – code và test xong, **chưa huấn luyện** (chờ dữ liệu Gate 1 của NV1).

## 1. Cách chạy

Chạy từ thư mục gốc của dự án, sau khi cài môi trường theo README gốc.

```bash
python -m pip install pytest
python -m pytest tests -k nv2 -v          # test hợp đồng (tensor ngẫu nhiên, không phải kết quả)
python -m scripts.train_cnn --config configs/cnn.yaml --data-root .
python -m scripts.eval_cnn --config configs/cnn.yaml --data-root . --split test
python -m scripts.infer_cnn duong_dan_anh.jpg
```

- `--data-root`: thư mục mà đường dẫn ảnh trong `data/splits/*.csv` được tính tương đối từ đó.
- `train_cnn` chỉ đọc Train và Validation. `eval_cnn --split test` chỉ chạy **một lần** sau khi đã chốt checkpoint.
- Đọc dữ liệu và transform dùng module của NV1 (`src/data/`); khi NV1 chưa bàn giao, script báo lỗi rõ ràng và dừng.

Đầu ra:

| File | Nội dung |
|---|---|
| `checkpoints/cnn_best.pt` | state_dict thuần của epoch có `val_macro_f1` cao nhất (demo NV4 nạp trực tiếp) |
| `checkpoints/cnn_best.meta.json` | nhãn, preprocessing, epoch, `val_macro_f1`, seed, config, commit git, môi trường |
| `results/cnn/train_history.csv` | loss/accuracy/macro F1 Train và Val theo epoch |
| `results/cnn/run_info.json` | cấu hình, môi trường, thời gian train, epoch tốt nhất |
| `figures/cnn_train_curves.png` | đường cong loss/accuracy |
| `results/cnn/cnn_metrics.json` | Accuracy, macro P/R/F1, per-class, confusion matrix |
| `results/cnn/cnn_test_predictions.csv` | dự đoán và xác suất từng ảnh Test |
| `results/cnn/cnn_confusion_matrix.png` | confusion matrix |

## 2. Kiến trúc

Mạng `TomatoCNN` (`src/models/cnn.py`), huấn luyện từ đầu, không dùng pretrained weights. Đầu vào ảnh RGB 128×128.

| Tầng | Cấu hình | Shape đầu ra | Số tham số |
|---|---|---|---|
| Đầu vào | ảnh RGB đã normalize | 3×128×128 | 0 |
| Khối 1 | Conv 3×3, 32 kênh, stride 1, padding 1, không bias → BatchNorm → ReLU → MaxPool 2×2 | 32×64×64 | 864 + 64 |
| Khối 2 | Conv 3×3, 64 kênh → BN → ReLU → MaxPool 2×2 | 64×32×32 | 18.432 + 128 |
| Khối 3 | Conv 3×3, 128 kênh → BN → ReLU → MaxPool 2×2 | 128×16×16 | 73.728 + 256 |
| Khối 4 | Conv 3×3, 256 kênh → BN → ReLU → MaxPool 2×2 | 256×8×8 | 294.912 + 512 |
| Global Average Pooling | lấy trung bình mỗi kênh trên 8×8 | 256 | 0 |
| Dropout | p = 0.3 (chỉ khi train) | 256 | 0 |
| Linear | 256 → 4 | 4 (logits) | 1.028 |

- **Tổng số tham số: 389.924** (kiểm tra tự động trong `tests/test_nv2_cnn_model.py`).
- Số tham số conv = `kênh_vào × kênh_ra × 3 × 3` (không bias vì BatchNorm đã có tham số dịch β). BatchNorm có 2 tham số học (γ, β) mỗi kênh. Linear = `256 × 4 + 4`.

Ý nghĩa từng thành phần:

- **Convolution 3×3**: trượt bộ lọc 3×3 trên ảnh để phát hiện đặc trưng cục bộ (cạnh, đốm, vân). Padding 1 giữ nguyên kích thước; stride 1 trượt từng pixel. Các khối sau nhìn vùng rộng hơn nên học đặc trưng phức tạp hơn (hình dạng vết bệnh).
- **BatchNorm**: chuẩn hóa đầu ra mỗi kênh theo batch, giúp train ổn định và nhanh hơn.
- **ReLU**: `max(0, x)`, tạo tính phi tuyến để mạng học được quan hệ phức tạp.
- **MaxPool 2×2**: giữ giá trị lớn nhất mỗi ô 2×2, giảm một nửa chiều rộng/cao, giảm tính toán và giúp chịu được dịch chuyển nhỏ.
- **Global Average Pooling**: thay cho flatten; giảm mạnh số tham số của tầng fully connected (256×4 thay vì 256×8×8×4), nên ít overfit hơn.
- **Dropout**: tắt ngẫu nhiên 30% nơ-ron khi train để giảm overfit; khi `model.eval()` thì không tắt.
- **Linear + softmax**: Linear cho 4 điểm (logits); softmax biến thành 4 xác suất cộng lại bằng 1. Xác suất này là độ tin của mô hình, không phải chẩn đoán chắc chắn.
- **Cross-entropy và backpropagation**: loss = `-log(xác suất của lớp đúng)`. Backpropagation tính đạo hàm của loss theo từng tham số; optimizer (Adam) cập nhật tham số theo hướng giảm loss.

## 3. Cấu hình huấn luyện

Giá trị đầy đủ trong `configs/cnn.yaml`. Đây là **đề xuất ban đầu**, sẽ kiểm chứng bằng Validation.

| Tham số           | Giá trị     | Lý do       |
|-------------------|-------------|-------------|
| Kích thước ảnh    | 128×128     | Mạng nhỏ, train nhanh; vẫn đủ thấy đốm bệnh |
| Normalize         | mean/std = 0.5 (tạm) | Chờ NV1 cung cấp mean/std tính trên Train |
| Loss              | CrossEntropyLoss | Chuẩn cho phân loại nhiều lớp |
| Optimizer         | Adam, weight decay 1e-4 | Hội tụ nhanh, ít phải chỉnh tay; weight decay giảm overfit |
| Learning rate     | 0.001 | Mặc định phổ biến của Adam; thử nghiệm E1 so với 0.0003 |
| LR scheduler      | không dùng | Giữ đơn giản cho baseline |
| Batch size        | 32 | Vừa bộ nhớ GPU 4 GB |
| Epochs            | tối đa 40 | Early stopping dừng sớm nếu Val không cải thiện |
| Early stopping    | patience 7 theo `val_macro_f1` | Tránh train tiếp khi đã overfit |
| Checkpoint policy | lưu epoch có `val_macro_f1` cao nhất | Macro F1 công bằng hơn Accuracy khi lớp lệch; giống tiêu chí của NV3 |
| Seed              | 42 | Tái lập kết quả; giống NV3 |

## 4. Môi trường

- Môi trường phát triển và test: Windows 11, Python 3.12.10, GPU NVIDIA RTX 3050 Laptop 4 GB.
- Môi trường huấn luyện thật: (chưa điền – ghi khi train; `run_info.json` tự lưu phiên bản Python/PyTorch/torchvision và thiết bị)

## 5. Thử nghiệm có kiểm soát (chỉ chọn bằng Validation)

(chưa điền)

## 6. Kết quả

Chỉ điền số liệu thực tế, truy vết được đến log và checkpoint.

- Biểu đồ loss/accuracy: `figures/cnn_train_curves.png` (chưa có)
- Metrics: (chưa có)
- Confusion matrix: (chưa có)

| Lớp            | Precision   | Recall      | F1          |
|----------------|-------------|-------------|-------------|
| Healthy        | (chưa điền) | (chưa điền) | (chưa điền) |
| Early Blight   | (chưa điền) | (chưa điền) | (chưa điền) |
| Late Blight    | (chưa điền) | (chưa điền) | (chưa điền) |
| Bacterial Spot | (chưa điền) | (chưa điền) | (chưa điền) |
| Macro avg      | (chưa điền) | (chưa điền) | (chưa điền) |

- Accuracy (Test): (chưa điền)

## 7. Phân tích

(chưa điền) – điểm mạnh, điểm yếu, các nhầm lẫn, dấu hiệu overfitting.

## Điều kiện nghiệm thu (mục IV.4)

- [ ] CNN huấn luyện TỪ ĐẦU, đầu ra đúng 4 lớp.
- [ ] Cùng split và mapping với NV1, không dùng Test để chọn model.
- [ ] Checkpoint chạy inference được trên máy khác theo README này.
- [ ] Có log/biểu đồ/metrics thực tế.
- [ ] Chỉ rõ điểm mạnh, điểm yếu, nhầm lẫn và dấu hiệu overfitting.
