# NV1 – dữ liệu PlantVillage (4 lớp cà chua)

**Phụ trách:** Lê Xuân Bảo · **Nguồn dữ liệu:** https://huggingface.co/datasets/mohanty/PlantVillage (bản `data.zip`, thư mục `raw/color/`).

**ZIP được lưu ngoài GitHub:** [Google Drive – ML_Tomato_Dataset/v1_manifests](https://drive.google.com/drive/folders/1gazFJNC1zdyAj8adjYeKTlOu1JJWIj7v), file `tomato_6502_images.zip` (~105.5 MB thập phân). Cần cấp quyền truy cập Drive cho người nhận. Không commit ZIP hay JPG vào GitHub.

## Data contract (đã xuất từ manifest thực tế)

- `data/dataset_manifest.csv`: 6.502 dòng, cột `split,path,label,group_id,leaf_map_matched,sha256`, giữ nguyên dữ liệu kiểm toán từ Colab.
- `data/splits/train.csv`: 4.115 dòng; `val.csv`: 1.031; `test.csv`: 1.356.
- Mỗi split CSV có `path,label,group`; `path` tương đối với thư mục giải nén, ví dụ `raw/color/Tomato___healthy/a.JPG`; `label` số nguyên `0,1,2,3`; `group` là `group_id` để kiểm toán.
- Mapping **không theo alphabet**: 0 Healthy; 1 Early Blight; 2 Late Blight; 3 Bacterial Spot (theo `configs/class_mapping.json`).
- Train/Val/Test là **cố định**, không tự chia lại và không dùng Test để tuning. Các ảnh gốc không bị tăng số file nhờ online augmentation.

## Chuẩn bị và kiểm tra từ repository root

1. Cài `python -m pip install -r requirements.txt` trong môi trường Python hỗ trợ PyTorch.
2. Tải `tomato_6502_images.zip` từ thư mục Drive trên về máy.
3. Chạy:

```bash
python -m scripts.prepare_dataset --archive /duong-dan/tomato_6502_images.zip --data-root data/raw
python -m scripts.check_dataset --data-root data/raw
python -m scripts.visualize_dataset --data-root data/raw
```

Sau giải nén, đường dẫn ảnh ví dụ là `data/raw/raw/color/Tomato___healthy/*.JPG` (do archive giữ cấu trúc `raw/color/...`). Đừng bỏ tầng `raw/` của ZIP.

## Sử dụng qua một API chung

```python
from src.data.dataset import create_dataloaders
loaders, datasets, class_to_idx = create_dataloaders(
    data_root="data/raw", model_name="cnn", batch_size=32,
    augment=True, seed=42,
)
images, labels = next(iter(loaders["train"]))
assert images.shape[1:] == (3, 224, 224)

# ResNet18: cùng split, khác resize/crop phù hợp pretrained.
r_loaders, r_datasets, _ = create_dataloaders(
    data_root="data/raw", model_name="resnet18", batch_size=32
)
```

Demo NV4 gọi `src.data.transforms.build_val_transform(model_name, config)`. Val/Test không có random transform. CNN: Resize 224×224; ResNet18: Resize cạnh ngắn 256 rồi CenterCrop 224; cả hai dùng ImageNet mean/std. NV3 PR #7 đang sử dụng một adapter CSV riêng tương thích `path,label`, nhưng cần xác nhận chạy thật trên bản split này.

## Giới hạn và nguồn gốc

- Tổng ảnh 4 lớp trước làm sạch: 6.627. Loại 14 bản trùng SHA-256 và 111 ảnh Train thuộc nhóm nghi ngờ chồng lấn Test, còn 6.502. Có nhật ký ở `data/audit/`.
- Không có SHA-256 hoặc group_id trùng giữa các tập theo quy tắc kiểm toán hiện tại; đây **không phải bằng chứng loại bỏ 100% hiện tượng trùng lá thực tế**.
- Trong giai đoạn kiểm toán ban đầu, 1.568 ảnh chưa khớp Leaf ID với metadata. Cách suy đoán nhóm theo tên file có giới hạn.
- PlantVillage chủ yếu là ảnh chụp có điều kiện kiểm soát; chưa suy diễn sang chẩn đoán thực địa/phát hiện sớm.
- Trước khi tái phân phối dữ liệu ra ngoài nhóm, cần kiểm tra quyền sử dụng/giấy phép tại nguồn Dataset và ảnh gốc.

Xem `docs/DATASET_README.md` cho thống kê và quyết định khoa học chi tiết.
