# NV3 — ResNet18 & Transfer Learning

**Phạm vi:** ResNet18 pretrained (ImageNet-1K V1), 4 tình trạng lá cà chua. Chủ nhiệm NV3 phụ trách mô hình; không thay đổi split/nhãn của NV1, CNN của NV2, app của NV4 hay phần tổng hợp NV5.

**Trạng thái:** Code và smoke tests có thể hoàn thiện trước khi NV1 bàn giao dữ liệu. **Chưa có huấn luyện thực nghiệm, checkpoint pretrained đã fine-tune hoặc kết quả Accuracy/F1 thực tế.** Không dùng test giả để báo cáo độ chính xác.

## Cấu trúc

- `src/models/resnet18.py`: backbone ResNet18, thay lớp `fc` từ 1000 sang 4, điều khiển freeze/unfreeze, BatchNorm.
- `src/resnet18_workflow.py`: adapter đọc `path,label` từ CSV, tiền xử lý riêng, mapping, checkpoint, inference, metrics.
- `scripts/train_resnet18.py`: Feature Extraction và Fine-tuning; chọn checkpoint dựa **Validation macro F1** (hòa thì xét Validation Loss).
- `scripts/eval_resnet18.py`: chỉ kiểm thử trên Test *sau khi đã chốt checkpoint*; confusion matrix và bảng chỉ số.
- `scripts/infer_resnet18.py`: đọc ảnh JPG/PNG, softmax 4 lớp đúng theo mapping.
- `configs/resnet18.yaml`: các tham số thử nghiệm ban đầu (chưa chứng minh tối ưu).
- `tests/test_nv3_resnet18.py`: kiểm tra cấu trúc, shape, freeze, dữ liệu, loss, checkpoint, metrics.

## Học thuật và kiến trúc

ResNet18 là CNN 18 lớp có residual/skip connections, giúp tối ưu hóa các mạng sâu. Các lớp Convolution trong backbone trích xuất đặc trưng; Global Average Pooling cung cấp **512 features**, `fc` mới ánh xạ **512 → 4** nên có **2.052 tham số trainable** khi backbone đóng băng. Tổng số tham số được đọc bằng `parameter_counts` sau khi tạo mô hình.

**Stage A (Feature Extraction):** khóa backbone, chỉ huấn luyện FC; BatchNorm bị khóa sử dụng running statistics đã học (ở `eval()`), không vô tình cập nhật thống kê.

**Stage B (Fine-tuning):** khởi tạo từ checkpoint tốt nhất Stage A, mở `layer4` và FC. LR backbone nhỏ hơn LR head. Chọn stage cuối bằng Validation, không dùng Test để chọn.

## Dữ liệu và hợp đồng với NV1

Đọc 3 manifest: `data/splits/train.csv`, `val.csv`, `test.csv`. Mỗi CSV có ít nhất cột `path,label`. `path` là đường dẫn tuyệt đối hoặc tương đối với `--data-root` (mặc định root repository). `label` có thể là số 0..3 hoặc tên lớp **khớp chính xác** với `configs/class_mapping.json`:

| ID | Nhãn |
|---|---|
| 0 | Healthy |
| 1 | Early Blight |
| 2 | Late Blight |
| 3 | Bacterial Spot |

Nếu manifest NV1 sau này có tên cột/đường dẫn khác, **trao đổi với NV1 trước khi sửa adapter**; không sửa các file do NV1 phụ trách và không tự tái chia dữ liệu. Lấy split gốc theo lá để tránh leakage theo quyết định NV1.

**Train transforms:** resize cạnh ngắn 256, random crop 224, horizontal flip, brightness/contrast jitter nhẹ (0.1), tensor, ImageNet normalization.

**Validation/Test/Inference:** resize cạnh ngắn 256, center crop 224, tensor, ImageNet normalization. Mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`. Center crop có thể bỏ sót tổn thương sát mép ảnh; đánh giá giới hạn trên ảnh thực địa.

## Môi trường và lệnh chạy

Ở thư mục root project, dùng Python hỗ trợ PyTorch, cài môi trường:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu cần CUDA, xem cấu hình tương thích tại https://pytorch.org/get-started/locally/ . Pretrained weights tải một lần qua mạng trong lần gọi `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`; nếu chưa có mạng hoặc cache thì không thể bắt đầu huấn luyện.

Chỉ sau khi NV1 hoàn tất Gate 1, xác nhận CSV và có ảnh thật:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_resnet18 --data-root .
.\.venv\Scripts\python.exe -m scripts.eval_resnet18 --data-root .
.\.venv\Scripts\python.exe -m scripts.infer_resnet18 "duong-dan-den-anh-la.jpg"
```

Kiểm thử kỹ thuật không cần tải pretrained weights hoặc ảnh thật:

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q tests/test_nv3_resnet18.py
```

Nếu thiếu CSV sẽ báo chờ `NV1 Gate 1` thay vì tự tạo split. Không chạy eval trên Test để chọn epochs/lr/checkpoint.

## Thông số thử nghiệm ban đầu

| Mục | Giá trị đề xuất |
|---|---|
| Weights | IMAGENET1K_V1 |
| Loss | CrossEntropyLoss (logits, không Softmax trước Loss) |
| Input | RGB 224×224 |
| Batch size / Seed | 32 / 42 |
| Feature extraction | AdamW; LR 0.001; 8 epochs tối đa |
| Fine-tuning | AdamW; layer4+fc; LR backbone 1e-5, head 1e-4; 10 epochs tối đa |
| Early stopping | Patience 4 ở fine-tuning |
| Checkpoint selection | Validation macro F1, tie-break Validation loss |

Các giá trị trên là cấu hình khởi đầu, **không phải kết quả đã xác minh tối ưu**. Ghi Python/torch/torchvision/CPU/GPU thật từ `results/resnet18/selected_stage.json` sau khi huấn luyện.

## File kết quả (sẽ xuất hiện sau khi chạy thật)

- `checkpoints/resnet18_best.pt`: model checkpoint tốt nhất theo Val.
- `results/resnet18/resnet18_feature_vs_finetune.csv`: Val macro F1, Val loss và parameter count hai stage.
- `results/resnet18/train_history.csv`, `figures/resnet18_train_curves.png`.
- `results/resnet18/resnet18_metrics.json`, `results/resnet18/resnet18_confusion_matrix.png`, `results/resnet18/resnet18_test_predictions.csv` (chỉ sau Test).

**NV4 integration:** `scripts.infer_resnet18.run(Path(image), Path(config))` trả `{"label": ..., "scores": {"Healthy": score, ...}}`; không mặc định điểm Softmax đã được hiệu chỉnh xác suất chẩn đoán.

**NV5 reporting:** Chỉ trích số liệu từ log/checkpoint/Test, so sánh trên cùng split với CNN của NV2. Không khẳng định phát hiện triệu chứng giai đoạn sớm nếu chưa có ảnh và đánh giá riêng.

## Checklist nghiệm thu

- [ ] Checkpoint đã được huấn luyện thật trên split chính thức và đủ 4 đầu ra.
- [ ] Có log chạy Feature Extraction và Fine-tuning thật.
- [ ] Không tái chia dữ liệu hoặc dùng Test để chọn cấu hình.
- [ ] Checkpoint đã tải lại và dự đoán đúng mapping/preprocessing.
- [ ] Có metrics, biểu đồ và phân tích giới hạn từ thực nghiệm.
