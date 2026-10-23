# NV1 – Dataset & Preprocessing (bàn giao Gate 1)

**Người phụ trách:** Lê Xuân Bảo · **Ngày:** 23/10/2026

## 1. Nguồn và phạm vi

- Nguồn: [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage), file `data.zip`, ảnh trong `raw/color/`.
- Chỉ 4 nhãn `Tomato___healthy`, `Tomato___Early_blight`, `Tomato___Late_blight`, `Tomato___Bacterial_spot`.
- Tất cả ảnh được khảo sát là JPEG, RGB, 256 × 256 px; không thấy lỗi đọc trong kiểm tra Pillow đã chạy trên Colab.
- Link ZIP và hướng dẫn: [data/README.md](../data/README.md). Ngày khảo sát/bàn giao 23/10/2026; ngày tải gốc cần đối chiếu log Colab khi ghi báo cáo.
- Kiểm tra giấy phép phân phối lại ở nguồn trước khi công bố ảnh ra ngoài nhóm.

## 2. Thống kê ảnh từ manifest chính thức

| Lớp | Ban đầu | Train | Val | Test | Được chọn |
|---|---:|---:|---:|---:|---:|
| Healthy | 1.591 | 944 | 240 | 330 | 1.514 |
| Early Blight | 1.000 | 632 | 153 | 215 | 1.000 |
| Late Blight | 1.909 | 1.188 | 298 | 375 | 1.861 |
| Bacterial Spot | 2.127 | 1.351 | 340 | 436 | 2.127 |
| **Tổng** | **6.627** | **4.115** | **1.031** | **1.356** | **6.502** |

Dữ liệu không cân bằng hoàn toàn: Bacterial Spot nhiều hơn Early Blight; báo cáo nên ưu tiên thêm macro-F1 và per-class F1 bên cạnh Accuracy. Chưa mặc định oversampling/class weights – chỉ áp dụng khi có thí nghiệm riêng trên Validation.

## 3. Kiểm toán và split

1. Kiểm tra dữ liệu gốc với Pillow: 0 file không đọc được.
2. Phát hiện 14 ảnh trùng SHA-256, loại dư thừa khỏi danh sách sử dụng.
3. Đối chiếu Leaf ID nguồn (một phần khớp metadata), dùng quy tắc nhóm theo tên file để phát hiện 90 nhóm nghi vấn qua Train/Test; loại 111 ảnh Train thuộc các nhóm nghi vấn.
4. Giữ Test gốc 1.356 ảnh; chia Validation từ Train đã làm sạch bằng `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, chọn fold gần tỷ lệ mong muốn.
5. Không có trùng đường dẫn, SHA-256 hoặc group_id giữa Train, Val, Test theo manifest cuối cùng. `group_id` bao gồm giá trị từ đối chiếu metadata và heuristic tên ảnh, **không đảm bảo tuyệt đối trùng lá vật lý đã được loại bỏ**.

Các chứng cứ nguồn: `data/dataset_manifest.csv`, `data/audit/removed_duplicates.csv`, `data/audit/excluded_train_leaf_overlap.csv`, `data/splits/*.csv`. Split CSV được chiết xuất trực tiếp từ manifest gốc **không thực hiện lại thuật toán chia**.

## 4. Preprocessing và tăng cường dữ liệu

- Nhãn cố định: `configs/class_mapping.json` với 0 Healthy, 1 Early Blight, 2 Late Blight, 3 Bacterial Spot.
- CNN: Resize (224,224) -> ToTensor -> Normalize, mô phỏng pipeline Colab đã kiểm thử.
- ResNet18: Resize cạnh ngắn 256 -> CenterCrop (224,224) -> ToTensor -> Normalize, phù hợp config và PR #7 NV3. Lưu ý: pipeline này khác bản baseline ResNet18 dùng trong Colab (Resize 224); phải huấn luyện/đánh giá nhất quán với pipeline chính thức sau khi nhóm xác nhận.
- Normalize chung ImageNet: mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
- CNN Train augmentation: HorizontalFlip 0.5; VerticalFlip 0.5; RandomRotation ±15°; ColorJitter brightness 0.10, contrast 0.10, saturation 0.08.
- ResNet18 Train augmentation (theo PR #7): RandomCrop 224 sau Resize256, HorizontalFlip 0.5, ColorJitter brightness/contrast 0.10.
- **Val/Test/Inference:** tất định, tuyệt đối không random augmentation.
- Đây là augmentation **online**; tổng số file ảnh vẫn là 6.502.

## 5. Giao diện bắt buộc cho thành viên

```python
# Train/Validation/Test: from repository root
from src.data.dataset import create_dataloaders
loaders, datasets, mapping = create_dataloaders(data_root="data/raw", model_name="cnn")

# NV4 Demo / NV2, NV3 inference
from src.data.transforms import build_val_transform
import yaml
config = yaml.safe_load(open("configs/cnn.yaml", encoding="utf-8"))
transform = build_val_transform("cnn", config)
```

`src.data.dataset.ManifestDataset(manifest, data_root, transform)` chấp nhận CSV `path,label[,group]`. Trong split, `path` tương đối từ `data_root`; mọi nhãn là số 0..3. Có thể sử dụng adapter `ManifestDataset` riêng của NV3 nếu dùng đúng split/transform.

## 6. Lệnh chạy và chứng cứ

```bash
python -m scripts.prepare_dataset --archive /path/tomato_6502_images.zip --data-root data/raw
python -m scripts.check_dataset --data-root data/raw
python -m scripts.visualize_dataset --data-root data/raw
```

`visualize_dataset.py` tạo `figures/class_distribution.png`, `figures/augmentation_examples.png` từ ảnh gốc đã giải nén, không đưa số liệu hoặc hình giả. Lưu hình kết quả vào PR khi được sinh ra.

**Trạng thái:** Colab trước đó đã PASS DataLoader và Clean Extraction với `tomato_data.py`. Mã chuẩn hóa mới ở `src/data/` cần được NV2 và NV3 smoke-test độc lập trên bộ ZIP này; không coi đó là PASS trước khi có log xác nhận.

## 7. Checklist Gate 1 trước khi merge

- [x] Có CSV và manifest 6.502 ảnh, mapping và số lượng từng lớp.
- [x] Có code Dataset, transforms, script kiểm toán/giải nén/trực quan hóa và README.
- [ ] Chạy script trên máy/môi trường sạch, đối chiếu 6.502 file ảnh thật.
- [ ] NV2 và NV3 mỗi người load một batch và xác nhận class IDs, tensor shape, sample paths.
- [ ] NV4 xác nhận `build_val_transform(model_name, config)` tương thích app Demo.
- [ ] Nhóm chốt chính thức chính sách crop/resize riêng CNN và ResNet18 trước khi huấn luyện.
- [ ] Đính kèm hình PNG, bằng chứng thực thi và phản hồi vào PR; reviewer phê duyệt.
