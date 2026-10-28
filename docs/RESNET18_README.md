# NV3 — ResNet18 & Transfer Learning

**Phạm vi:** ResNet18 pretrained (ImageNet-1K V1), 4 tình trạng lá cà chua. Chủ nhiệm NV3 phụ trách mô hình; không thay đổi split/nhãn của NV1, CNN của NV2, app của NV4 hay phần tổng hợp NV5.

**Trạng thái:** Đã sửa theo review PR #7: factory offline, checkpoint thuần + metadata JSON, dùng chung API NV1. Nhánh NV3 cần tích hợp NV1 PR #8 trước khi chạy luồng dữ liệu. **Chưa có huấn luyện thực nghiệm, checkpoint pretrained đã fine-tune hoặc kết quả Accuracy/F1 thực tế.** Không dùng test giả để báo cáo độ chính xác.

## Cấu trúc

- `src/models/resnet18.py`: backbone ResNet18, thay lớp `fc` từ 1000 sang 4, điều khiển freeze/unfreeze, BatchNorm.
- `src/resnet18_workflow.py`: wrapper gọi Dataset/transforms dùng chung của NV1, mapping, checkpoint, inference, metrics.
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

Đọc 3 manifest: `data/splits/train.csv`, `val.csv`, `test.csv`. Mỗi CSV có ít nhất cột `path,label`. `path` là đường dẫn tương đối an toàn với `--data-root` (CLI mặc định `data/raw`). `label` có thể là số 0..3 hoặc tên lớp **khớp chính xác** với `configs/class_mapping.json`:

| ID | Nhãn |
|---|---|
| 0 | Healthy |
| 1 | Early Blight |
| 2 | Late Blight |
| 3 | Bacterial Spot |

Đã đối chiếu NV1 commit `6a03cbe`: `path,label,group`, đường dẫn `raw/color/...`; ảnh sau giải nén nằm ở `data/raw/raw/color/...`. Dataset kiểm tra nhãn khớp thư mục lớp và từ chối đường dẫn không an toàn. Không hardcode số ảnh. Nếu manifest NV1 sau này có tên cột/đường dẫn khác, **trao đổi với NV1 trước khi sửa adapter**; không sửa các file do NV1 phụ trách và không tự tái chia dữ liệu. Lấy split gốc theo lá để tránh leakage theo quyết định NV1.

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
.\.venv\Scripts\python.exe -m scripts.train_resnet18 --data-root data/raw
.\.venv\Scripts\python.exe -m scripts.eval_resnet18 --data-root data/raw
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

## Hợp đồng checkpoint NV4

```python
from src.models.resnet18 import build_model
model = build_model(num_classes=4)  # không truy cập Internet
state = torch.load("checkpoints/resnet18_best.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
```

Checkpoint mới là plain `state_dict`; `resnet18_best.meta.json` đi kèm chứa mapping, preprocessing, ImageNet weights version, stage, epoch, val_macro_f1, val_loss, seed, toàn bộ config và phiên bản môi trường. Các checkpoint từng stage cũng dùng định dạng này. Loader NV3 vẫn đọc được checkpoint cũ dạng `model_state_dict` + metadata; không tự chuyển đổi hay xóa file cũ. CLI NV3 yêu cầu sidecar cho checkpoint thuần để kiểm tra mapping/preprocessing; loader NV4 trực tiếp tải state_dict.

Mỗi lần chạy mới cần đường dẫn checkpoint/results chưa chứa checkpoint cũ: chương trình từ chối ghi đè để bảo toàn trọng số. Nếu đổi augmentation, cập nhật pipeline chung qua NV1; wrapper NV3 không còn giữ augmentation riêng.

## Kiểm chứng sửa review (28/10/2026)

- Python 3.12.14, torch 2.14.1+cpu, torchvision 0.29.1+cpu.
- Kết quả: **8 pass, 4 skip** trên nhánh NV3; **12/12 pass** trên bản sao tích hợp NV1/NV3/NV4. Các test dữ liệu/tích hợp được đánh dấu skip nếu chưa tích hợp API NV1/NV4; không coi skip là pass.
- Bản sao tạm tích hợp code NV1 `6a03cbe`, NV4 `21b1778` và code NV3 hiện tại: kiểm tra dataset thật của NV1 bằng ảnh tổng hợp đúng schema; hai stage, checkpoint, eval, inference; xác suất NV3 khớp API NV4 trên cùng ảnh.
- Đã tải ImageNet weights chính thức qua torchvision (kiểm tra checksum), forward được `[1,4]`. Dùng `TORCH_HOME=/workspace/.cache/torch` trong môi trường cloud để tái dùng cache.
- Không có bộ ảnh PlantVillage trong môi trường; chưa chạy huấn luyện chính thức, chưa đánh giá Test thật, chưa nghiệm thu Gate 1. Dữ liệu tổng hợp chỉ phục vụ kiểm thử kỹ thuật.
- Linux: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q tests/test_nv3_resnet18.py`. Windows PowerShell dùng lệnh pytest ở trên. Tích hợp PR #8/#1 là phần việc phối hợp; không tự sửa file NV1/NV4 hoặc merge main.

**NV4 integration:** `scripts.infer_resnet18.run(Path(image), Path(config))` trả `{"label": ..., "scores": {"Healthy": score, ...}}`; không mặc định điểm Softmax đã được hiệu chỉnh xác suất chẩn đoán.

**NV5 reporting:** Chỉ trích số liệu từ log/checkpoint/Test, so sánh trên cùng split với CNN của NV2. Không khẳng định phát hiện triệu chứng giai đoạn sớm nếu chưa có ảnh và đánh giá riêng.

## Checklist nghiệm thu

- [ ] Checkpoint đã được huấn luyện thật trên split chính thức và đủ 4 đầu ra.
- [ ] Có log chạy Feature Extraction và Fine-tuning thật.
- [ ] Không tái chia dữ liệu hoặc dùng Test để chọn cấu hình.
- [ ] Checkpoint đã tải lại và dự đoán đúng mapping/preprocessing.
- [ ] Có metrics, biểu đồ và phân tích giới hạn từ thực nghiệm.


## NV3-01 — kiểm chứng tích hợp

Đã kiểm tra lại remote/working tree, PR #7/#8 và các nhánh ngày 28/10/2026. Code contract có từ commit `3412da6`; lần NV3-01 bổ sung test trực tiếp API `create_dataloaders` NV1 và sửa ví dụ `--data-root data/raw` trong docstring script train.

Các nguồn được kiểm chứng: NV1 `6a03cbe762aef571d56f36805665a253ad2af87a`, NV4 `21b177867c4196a771f02813ba5d2c4f7af3a7a7`. Bản sao ngoài checkout được tạo bằng `git archive`, chỉ ghép file dữ liệu/manifest NV1 và API inference NV4; không merge nhánh, không sửa file thành viên khác. Lệnh kiểm thử:

```bash
# Từ root checkout NV3:
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/dudoan-mpl .venv/bin/python -m pytest -q tests/test_nv3_resnet18.py
# 8 passed, 4 skipped: thiếu code NV1/NV4 trong nhánh NV3.

# Từ /workspace/onboarding/nv3-01-integration (bản sao cô lập):
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/dudoan-mpl /workspace/Project_Dudoanbenhcaycachua/.venv/bin/python -m pytest -q tests/test_nv3_resnet18.py
# 12 passed, 0 failed, 0 skipped.
```

Test mới xác minh manifest `path,label,group`, mapping 0..3, ảnh grayscale được Dataset chuyển RGB, batch `[4,3,224,224]`, giá trị ImageNet Normalize, Val tất định và tensor Val NV3 trùng API NV1. Test luồng training chặn mọi truy cập Dataset Test trong lúc chọn mô hình; Test tổng hợp chỉ được đọc ở bước eval sau training. Năm test gốc vẫn chạy đạt trong bản sao tích hợp. Factory không tải mạng, freeze/unfreeze/BatchNorm, strict checkpoint load, legacy metadata và xác suất NV4 khớp NV3 đều đã được kiểm tra.

Không có ảnh thật: chưa xác minh checksum ảnh PlantVillage, chưa nghiệm thu Gate 1, chưa huấn luyện chính thức hoặc có checkpoint bệnh lá bàn giao. Smoke checkpoints chỉ nằm trong thư mục tạm của pytest, không được commit hoặc đưa vào demo như trọng số thật. PR #8 cần được tích hợp qua quy trình review của nhóm trước khi chạy luồng dữ liệu trên nhánh chung.
