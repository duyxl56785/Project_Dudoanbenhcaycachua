# CNN tự xây dựng

Phụ trách: NV2 - Nguyễn Văn Duy. Trạng thái: Sẵn sàng review – đã huấn luyện, chọn cấu hình bằng Validation và đánh giá Test một lần (28/10/2026).

## 1. Cách chạy

Chạy từ thư mục gốc của dự án, sau khi cài môi trường theo README gốc.

```bash
python -m pip install pytest
python -m pytest tests -k nv2 -v          # test hợp đồng (tensor ngẫu nhiên, không phải kết quả)
python -m scripts.train_cnn --config configs/cnn.yaml --data-root data/raw --run-name e1_lr3e-4
python -m scripts.eval_cnn --config configs/cnn.yaml --data-root data/raw --split test
python -m scripts.infer_cnn duong_dan_anh.jpg
```

- Dữ liệu: giải nén `tomato_6502_images.zip` theo [data/README.md](../data/README.md) vào `data/raw`.
- `--data-root`: thư mục mà đường dẫn ảnh trong `data/splits/*.csv` được tính tương đối từ đó (`data/raw`).
- Ba run đã chạy: `--run-name baseline --learning-rate 0.001`, `--run-name e1_lr3e-4` (giá trị hiện tại của config), `--run-name e2_no_aug --learning-rate 0.001 --no-augmentation`.
- `train_cnn` chỉ đọc Train và Validation. `eval_cnn --split test` chỉ chạy **một lần** sau khi đã chốt checkpoint.
- Đọc dữ liệu và transform dùng module của NV1 (`src/data/`); khi NV1 chưa bàn giao, script báo lỗi rõ ràng và dừng.

Đầu ra:

| File | Nội dung |
|---|---|
| `checkpoints/cnn_best.pt` | state_dict thuần của epoch có `val_macro_f1` cao nhất (demo NV4 nạp trực tiếp) |
| `checkpoints/cnn_best.meta.json` | nhãn, preprocessing, epoch, `val_macro_f1`, seed, config, commit git, môi trường |
| `results/cnn/runs/<run>/` | mỗi run: `train_history.csv` (loss/accuracy/macro F1 Train và Val theo epoch), `run_info.json` (cấu hình, môi trường, thời gian, epoch tốt nhất), `cnn_best.pt`, `cnn_train_curves.png` |
| `results/cnn/experiments.csv` | bảng so sánh các run trên Validation |
| `figures/cnn_train_curves.png` | đường cong loss/accuracy của run được chọn |
| `results/cnn/cnn_metrics.json` | Accuracy, macro P/R/F1, per-class, confusion matrix |
| `results/cnn/cnn_test_predictions.csv` | dự đoán và xác suất từng ảnh Test |
| `results/cnn/cnn_confusion_matrix.png` | confusion matrix |

## 2. Kiến trúc

Mạng `TomatoCNN` (`src/models/cnn.py`), huấn luyện từ đầu, không dùng pretrained weights. Đầu vào ảnh RGB 224×224 (chuẩn chung của nhóm). Nhờ Global Average Pooling, số tham số không phụ thuộc kích thước ảnh.

| Tầng | Cấu hình | Shape đầu ra | Số tham số |
|---|---|---|---|
| Đầu vào | ảnh RGB đã normalize | 3×224×224 | 0 |
| Khối 1 | Conv 3×3, 32 kênh, stride 1, padding 1, không bias → BatchNorm → ReLU → MaxPool 2×2 | 32×112×112 | 864 + 64 |
| Khối 2 | Conv 3×3, 64 kênh → BN → ReLU → MaxPool 2×2 | 64×56×56 | 18.432 + 128 |
| Khối 3 | Conv 3×3, 128 kênh → BN → ReLU → MaxPool 2×2 | 128×28×28 | 73.728 + 256 |
| Khối 4 | Conv 3×3, 256 kênh → BN → ReLU → MaxPool 2×2 | 256×14×14 | 294.912 + 512 |
| Global Average Pooling | lấy trung bình mỗi kênh trên 14×14 | 256 | 0 |
| Dropout | p = 0.3 (chỉ khi train) | 256 | 0 |
| Linear | 256 → 4 | 4 (logits) | 1.028 |

- **Tổng số tham số: 389.924** (kiểm tra tự động trong `tests/test_nv2_cnn_model.py`).
- Số tham số conv = `kênh_vào × kênh_ra × 3 × 3` (không bias vì BatchNorm đã có tham số dịch β). BatchNorm có 2 tham số học (γ, β) mỗi kênh. Linear = `256 × 4 + 4`.

Ý nghĩa từng thành phần:

- **Convolution 3×3**: trượt bộ lọc 3×3 trên ảnh để phát hiện đặc trưng cục bộ (cạnh, đốm, vân). Padding 1 giữ nguyên kích thước; stride 1 trượt từng pixel. Các khối sau nhìn vùng rộng hơn nên học đặc trưng phức tạp hơn (hình dạng vết bệnh).
- **BatchNorm**: chuẩn hóa đầu ra mỗi kênh theo batch, giúp train ổn định và nhanh hơn.
- **ReLU**: `max(0, x)`, tạo tính phi tuyến để mạng học được quan hệ phức tạp.
- **MaxPool 2×2**: giữ giá trị lớn nhất mỗi ô 2×2, giảm một nửa chiều rộng/cao, giảm tính toán và giúp chịu được dịch chuyển nhỏ.
- **Global Average Pooling**: thay cho flatten; giảm mạnh số tham số của tầng fully connected (256×4 thay vì 256×14×14×4), nên ít overfit hơn.
- **Dropout**: tắt ngẫu nhiên 30% nơ-ron khi train để giảm overfit; khi `model.eval()` thì không tắt.
- **Linear + softmax**: Linear cho 4 điểm (logits); softmax biến thành 4 xác suất cộng lại bằng 1. Xác suất này là độ tin của mô hình, không phải chẩn đoán chắc chắn.
- **Cross-entropy và backpropagation**: loss = `-log(xác suất của lớp đúng)`. Backpropagation tính đạo hàm của loss theo từng tham số; optimizer (Adam) cập nhật tham số theo hướng giảm loss.

## 3. Cấu hình huấn luyện

Giá trị đầy đủ trong `configs/cnn.yaml`. Đây là cấu hình **cuối**, chọn bằng Validation (mục 5); chưa phải giá trị tối ưu toàn cục.

| Tham số           | Giá trị     | Lý do       |
|-------------------|-------------|-------------|
| Kích thước ảnh    | 224×224     | Chuẩn chung nhóm (NV1 đề xuất): cùng kích thước với ResNet18, dùng chung transform và demo |
| Normalize         | mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225] (ImageNet) | Chuẩn chung nhóm; CNN không dùng pretrained nên đây chỉ là bước đưa pixel về khoảng quanh 0 |
| Loss              | CrossEntropyLoss | Chuẩn cho phân loại nhiều lớp |
| Optimizer         | Adam, weight decay 1e-4 | Hội tụ nhanh, ít phải chỉnh tay; weight decay giảm overfit |
| Learning rate     | 0.0003 | E1 cho `val_macro_f1` cao nhất; baseline 0.001 là mặc định phổ biến của Adam (mục 5) |
| LR scheduler      | không dùng | Giữ đơn giản cho baseline |
| Batch size        | 32 | Đo trên RTX 3050 4 GB: đỉnh khoảng 1,1 GB VRAM mỗi bước train ở 224×224 |
| Epochs            | tối đa 40 | Early stopping dừng sớm nếu Val không cải thiện |
| Early stopping    | patience 7 theo `val_macro_f1` | Tránh train tiếp khi đã overfit |
| Checkpoint policy | lưu epoch có `val_macro_f1` cao nhất; bằng nhau thì chọn Val Loss thấp hơn | Macro F1 công bằng hơn Accuracy khi lớp lệch; theo chuẩn chung của nhóm (giống NV3) |
| Seed              | 42 | Tái lập kết quả; giống NV3 |

## 4. Môi trường

- Môi trường phát triển và test: Windows 11, Python 3.12.10, GPU NVIDIA RTX 3050 Laptop 4 GB.
- Môi trường huấn luyện thật (ghi trong `run_info.json`): Windows 11, Python 3.12.10, PyTorch 2.14.1+cu126, torchvision 0.29.1+cu126, GPU NVIDIA GeForce RTX 3050 Laptop (4 GB), `num_workers: 0`.
- Code: commit `b0b6fe3` (nhánh `feature/nv2-cnn`, đã gồm dữ liệu NV1 từ PR #8). Dữ liệu: `tomato_6502_images.zip`, split cố định Train 4.115 / Val 1.031 / Test 1.356.

## 5. Thử nghiệm có kiểm soát (chỉ chọn bằng Validation)

Mỗi run chỉ đổi **một** yếu tố so với baseline; cùng seed 42, cùng split, cùng kiến trúc. Số liệu là của epoch tốt nhất theo `val_macro_f1` (nguồn: `results/cnn/experiments.csv`, `results/cnn/runs/<run>/train_history.csv`).

| Run | Thay đổi | Epoch tốt nhất / đã chạy | Val macro-F1 | Val accuracy | Val loss | Thời gian train |
|---|---|---|---|---|---|---|
| `baseline` | lr 1e-3, có augmentation | 22 / 29 | 0,9690 | 0,9728 | 0,0823 | 1.487,5 s |
| **`e1_lr3e-4`** | **lr 3e-4** | **24 / 31** | **0,9768** | **0,9806** | **0,0647** | 1.118,7 s |
| `e2_no_aug` | tắt augmentation Train | 11 / 18 | 0,9615 | 0,9651 | 0,1074 | 434,4 s |

**Cấu hình được chọn:** `e1_lr3e-4` (Val macro-F1 cao nhất). Checkpoint, đường cong và bảng so sánh được commit ở `7731e26` **trước** khi chạy Test.

## 6. Kết quả trên Test

Đánh giá **một lần** trên 1.356 ảnh Test, checkpoint `e1_lr3e-4` epoch 24 (nguồn: `results/cnn/cnn_metrics.json`, `results/cnn/cnn_test_predictions.csv`).

- Biểu đồ loss/accuracy: `figures/cnn_train_curves.png`
- Confusion matrix: `results/cnn/cnn_confusion_matrix.png`

| Lớp            | Precision | Recall | F1     | Số ảnh |
|----------------|-----------|--------|--------|--------|
| Healthy        | 0,9850    | 0,9970 | 0,9910 | 330    |
| Early Blight   | 0,9340    | 0,9209 | 0,9274 | 215    |
| Late Blight    | 0,9496    | 0,9547 | 0,9521 | 375    |
| Bacterial Spot | 0,9931    | 0,9862 | 0,9896 | 436    |
| Macro avg      | 0,9654    | 0,9647 | 0,9650 | 1.356  |

- Accuracy (Test): 0,9698 (1.315/1.356 ảnh đúng, 41 ảnh sai).

Confusion matrix (hàng = thực tế, cột = dự đoán; thứ tự Healthy, Early Blight, Late Blight, Bacterial Spot):

```
Healthy         329    0    1    0
Early Blight      0  198   14    3
Late Blight       5   12  358    0
Bacterial Spot    0    2    4  430
```

## 7. Phân tích

- **Điểm mạnh:** Healthy (F1 0,991) và Bacterial Spot (F1 0,990) gần như không bị nhầm. Mô hình nhỏ (389.924 tham số), mỗi run train dưới 25 phút trên GPU laptop.
- **Điểm yếu / nhầm lẫn chính:** cặp **Early Blight ↔ Late Blight** chiếm 26/41 lỗi (14 Early→Late, 12 Late→Early). Early Blight có F1 thấp nhất (0,927) và cũng là lớp ít ảnh Train nhất (632). Khi xem 10 ảnh sai có độ tin cao nhất: các lá có mảng hoại tử sẫm màu lớn bị nhầm qua lại giữa hai lớp blight; vài lá Late Blight chỉ có vết nhỏ ở mép hoặc vàng lá bị dự đoán Healthy (5 ảnh). Đây là quan sát bằng mắt, không phải đánh giá chuyên môn bệnh cây.
- **Độ tin chưa hiệu chỉnh:** có ảnh sai với xác suất softmax tới 0,97, nên xác suất không được hiểu là mức chắc chắn của chẩn đoán.
- **Overfitting:** chưa thấy dấu hiệu rõ ở run được chọn. Val loss giảm và đi ngang cùng Train loss; Val loss thấp hơn Train loss vì số liệu Train đo khi augmentation và dropout đang bật. Test macro-F1 (0,9650) thấp hơn Val (0,9768) khoảng 0,012.
- **Augmentation:** bỏ augmentation (E2) cho Val macro-F1 thấp nhất và Val loss dao động mạnh hơn ở các epoch cuối (0,099 → 0,149 → 0,088 → 0,129), phù hợp với vai trò giảm overfit của augmentation; chỉ là một thử nghiệm với một seed.
- **Giới hạn:** kết quả trên ảnh PlantVillage chụp trong điều kiện kiểm soát; chưa đánh giá ảnh thực địa hay bệnh giai đoạn sớm. Mỗi cấu hình mới chạy một seed.

## Điều kiện nghiệm thu (mục IV.4)

- [x] CNN huấn luyện TỪ ĐẦU, đầu ra đúng 4 lớp.
- [x] Cùng split và mapping với NV1, không dùng Test để chọn model (chọn ở `7731e26`, Test chạy sau).
- [ ] Checkpoint chạy inference được trên máy khác theo README này (mới chạy trên máy NV2; cần một thành viên khác xác nhận).
- [x] Có log/biểu đồ/metrics thực tế.
- [x] Chỉ rõ điểm mạnh, điểm yếu, nhầm lẫn và dấu hiệu overfitting.
