# NV1 – Dataset & Preprocessing (Bàn giao Gate 1)

**Người phụ trách:** Lê Xuân Bảo

**Ngày bàn giao ban đầu:** 23/10/2026

**Ngày cập nhật tài liệu:** 28/10/2026

**Issue:** #2 – Dataset & Preprocessing

**Pull Request:** [PR #8](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/8)

---
## 1. Nguồn và phạm vi dữ liệu

- **Nguồn dữ liệu:** [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage).
- **Dữ liệu gốc:** `data.zip`, các ảnh nằm trong `raw/color/`.
- **Phạm vi:** Chỉ sử dụng 4 lớp bệnh/tình trạng lá cà chua:
  1. `Tomato___healthy`
  2. `Tomato___Early_blight`
  3. `Tomato___Late_blight`
  4. `Tomato___Bacterial_spot`

Theo kết quả khảo sát dữ liệu đã thực hiện trên Google Colab:

- Các ảnh được khảo sát là JPEG, RGB, kích thước 256 × 256 pixel.
- Không phát hiện file ảnh không đọc được trong quá trình kiểm tra bằng Pillow.
- Tổng số ảnh ban đầu thuộc 4 lớp là **6.627 ảnh**.

Link ZIP và hướng dẫn sử dụng được lưu tại [data/README.md](../data/README.md).

**Lưu ý về nguồn:**

- Ngày khảo sát và bàn giao ban đầu: 23/10/2026.
- Ngày tải dữ liệu gốc cần được đối chiếu với log Colab nếu đưa vào báo cáo chính thức.
- Cần kiểm tra điều kiện giấy phép của nguồn dữ liệu trước khi phân phối ảnh ra ngoài nhóm.

---

## 2. Thống kê dữ liệu chính thức

Thống kê từ `data/dataset_manifest.csv` và `data/dataset_summary.csv`:

| Lớp | Ban đầu | Train | Val | Test | Được chọn |
|---|---:|---:|---:|---:|---:|
| Healthy | 1.591 | 944 | 240 | 330 | 1.514 |
| Early Blight | 1.000 | 632 | 153 | 215 | 1.000 |
| Late Blight | 1.909 | 1.188 | 298 | 375 | 1.861 |
| Bacterial Spot | 2.127 | 1.351 | 340 | 436 | 2.127 |
| **Tổng** | **6.627** | **4.115** | **1.031** | **1.356** | **6.502** |

### 2.1. Phân chia tập dữ liệu

- **Train:** 4.115 ảnh.
- **Validation:** 1.031 ảnh.
- **Test:** 1.356 ảnh.
- **Tổng dữ liệu sử dụng:** 6.502 ảnh.

Các tập dữ liệu đã được cố định tại:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

Các file CSV được xuất từ manifest chính thức. Script kiểm tra không thực hiện lại việc chia tập dữ liệu.

### 2.2. Phân bố lớp

Dữ liệu không cân bằng hoàn toàn giữa các lớp. Số ảnh Bacterial Spot lớn hơn số ảnh Early Blight.

Do đó, khi đánh giá mô hình, cần báo cáo thêm:

- Accuracy.
- Macro-F1.
- Precision, Recall và F1 theo từng lớp.
- Confusion Matrix.

Không mặc định sử dụng oversampling hoặc class weights khi chưa có thí nghiệm và đánh giá trên Validation.

---

## 3. Kiểm toán, làm sạch và chia dữ liệu

### 3.1. Kiểm tra dữ liệu gốc

Trong quá trình tiền xử lý ban đầu:

1. Kiểm tra khả năng đọc ảnh bằng Pillow.
2. Không phát hiện file ảnh không đọc được.
3. Sử dụng SHA-256 để phát hiện ảnh trùng nội dung.
4. Phát hiện và loại khỏi danh sách sử dụng 14 ảnh trùng SHA-256.
5. Đối chiếu Leaf ID nguồn khi có metadata phù hợp.
6. Áp dụng quy tắc nhóm dựa trên tên file để phát hiện các trường hợp nghi ngờ cùng lá vật lý.

### 3.2. Xử lý nguy cơ rò rỉ dữ liệu

Kết quả kiểm toán ban đầu ghi nhận:

- 90 nhóm nghi vấn xuất hiện qua Train/Test.
- Loại 111 ảnh Train thuộc các nhóm nghi vấn.
- Giữ nguyên Test gốc gồm 1.356 ảnh.
- Chia Validation từ Train đã làm sạch bằng:

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`

Fold được lựa chọn dựa trên mức độ phù hợp với tỷ lệ chia mong muốn.

### 3.3. Kiểm tra tính độc lập giữa các tập

Theo manifest cuối cùng:

- Không có đường dẫn ảnh trùng nhau giữa Train, Val và Test.
- Không có SHA-256 trùng nhau giữa các tập.
- Không có `group_id` trùng nhau giữa các tập.

**Giới hạn cần lưu ý:**

`group_id` được tạo từ metadata đối chiếu được và heuristic dựa trên tên ảnh.

Vì vậy, kết quả kiểm toán không chứng minh tuyệt đối rằng mọi trường hợp cùng lá vật lý đã được loại bỏ.

Không được mô tả phương pháp này là bảo đảm 100% không có data leakage.

### 3.4. Các file bằng chứng

- `data/dataset_manifest.csv`
- `data/dataset_summary.csv`
- `data/audit/removed_duplicates.csv`
- `data/audit/excluded_train_leaf_overlap.csv`
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

---

## 4. Preprocessing và Data Augmentation

### 4.1. Thứ tự nhãn

Sử dụng mapping cố định trong `configs/class_mapping.json`:

| Label ID | Tên lớp |
|---|---|
| 0 | Healthy |
| 1 | Early Blight |
| 2 | Late Blight |
| 3 | Bacterial Spot |

Không tự động sắp xếp lại thứ tự nhãn theo alphabet.

### 4.2. Preprocessing cho CNN

Pipeline Validation/Test/Inference:

1. Resize ảnh về 224 × 224.
2. Chuyển ảnh thành tensor.
3. Normalize theo ImageNet.

Đây là pipeline CNN đã được dùng trong quá trình kiểm thử tích hợp NV1–NV2.

### 4.3. Preprocessing cho ResNet18

Pipeline Validation/Test/Inference:

1. Resize cạnh ngắn về 256 pixel.
2. CenterCrop kích thước 224 × 224.
3. Chuyển ảnh thành tensor.
4. Normalize theo ImageNet.

Pipeline này phù hợp với cấu hình ResNet18 được đề xuất trong PR #7.

**Lưu ý:** Pipeline ResNet18 trong repository khác baseline ResNet18 trước đó trên Colab sử dụng Resize 224. Khi huấn luyện chính thức, nhóm phải thống nhất và sử dụng cùng một pipeline tương ứng cho Train/Validation/Test/Inference.

### 4.4. Tham số Normalize

Sử dụng giá trị ImageNet:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

### 4.5. Data Augmentation

**CNN – Train:**

- RandomHorizontalFlip với xác suất 0.5.
- RandomVerticalFlip với xác suất 0.5.
- RandomRotation trong khoảng ±15 độ.
- ColorJitter:
  - Brightness: 0.10.
  - Contrast: 0.10.
  - Saturation: 0.08.

**ResNet18 – Train:**

Theo cấu hình được đề xuất trong PR #7:

- Resize cạnh ngắn về 256.
- RandomCrop 224 × 224.
- RandomHorizontalFlip với xác suất 0.5.
- ColorJitter brightness/contrast 0.10.

**Nguyên tắc bắt buộc:**

- Augmentation chỉ được áp dụng cho Train.
- Validation, Test và Inference phải có preprocessing tất định.
- Không thực hiện random augmentation trên Test.
- Augmentation thực hiện online, không tạo thêm ảnh cố định trong Dataset.
- Tổng số ảnh sử dụng vẫn là 6.502.

---

## 5. Giao diện dữ liệu bàn giao cho các thành viên

### 5.1. Tạo DataLoader

Chạy từ thư mục gốc repository:

```python
from src.data.dataset import create_dataloaders

loaders, datasets, mapping = create_dataloaders(
    data_root="data/raw",
    model_name="cnn"
)
```

Với ResNet18:

```python
from src.data.dataset import create_dataloaders

loaders, datasets, mapping = create_dataloaders(
    data_root="data/raw",
    model_name="resnet18"
)
```

### 5.2. Sử dụng Validation Transform

Ví dụ với CNN:

```python
import yaml
from src.data.transforms import build_val_transform

with open("configs/cnn.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

transform = build_val_transform("cnn", config)
```

`configs/cnn.yaml` thuộc phạm vi phụ trách của NV2. PR #8 không thay đổi file này.

### 5.3. Cấu trúc ManifestDataset

Giao diện:

```python
from src.data.dataset import ManifestDataset
```

`ManifestDataset(manifest, data_root, transform)` nhận dữ liệu CSV với các cột:

- `path`
- `label`
- `group` (nếu có)

Trong đó:

- `path` tương đối so với `data_root`.
- `label` sử dụng ID từ 0 đến 3.
- `data_root="data/raw"` khi giải nén ZIP đúng cấu trúc dự án.

Các thành viên phải sử dụng cùng split và class mapping đã được cố định.

Nếu sử dụng adapter riêng, cần bảo đảm kết quả tương thích với giao diện dữ liệu chung.

---

## 6. Hướng dẫn chuẩn bị và kiểm tra dữ liệu

### 6.1. Chuẩn bị dữ liệu thật

Tải file `tomato_6502_images.zip` từ nguồn lưu trữ dữ liệu của NV1 theo hướng dẫn trong `data/README.md`.

Sau đó giải nén:

```powershell
python -m scripts.prepare_dataset --archive "duong_dan_den_zip/tomato_6502_images.zip" --data-root data/raw
```

Thay đường dẫn ZIP trong ví dụ bằng đường dẫn thực tế trên máy.

Không commit trực tiếp 6.502 ảnh vào GitHub khi chưa có quyết định chung về lưu trữ và quyền phân phối.

### 6.2. Kiểm tra metadata và split

```powershell
python -m scripts.check_dataset
```

Chức năng:

- Kiểm tra tổng số dòng manifest.
- Kiểm tra nhãn và số lượng ảnh.
- Kiểm tra đường dẫn, SHA-256 và group ID trong metadata.
- Phát hiện trùng đường dẫn hoặc hash trong manifest.
- Phát hiện trùng nhóm giữa các tập.
- Đối chiếu các split CSV với manifest gốc.

**Giới hạn:** Lệnh mặc định không đọc nội dung file ảnh để tính lại SHA-256.

### 6.3. Kiểm tra file ảnh tồn tại

```powershell
python -m scripts.check_dataset --data-root data/raw
```

Lệnh này kiểm tra sự tồn tại của các ảnh được tham chiếu trong manifest.

Nó không tính lại hash thực tế.

### 6.4. Kiểm tra SHA-256 trực tiếp từ ảnh

```powershell
python -m scripts.check_dataset --data-root data/raw --verify-hashes
```

Khi bật `--verify-hashes`, script:

1. Đọc từng ảnh thực tế.
2. Tính SHA-256 từ nội dung file bằng `hashlib`.
3. Đọc file theo từng khối 1 MiB để tránh nạp toàn bộ file vào bộ nhớ.
4. So sánh hash tính được với cột `sha256` trong manifest.
5. Báo lỗi nếu không tìm thấy ảnh hoặc nội dung ảnh không khớp hash.
6. Chỉ báo PASS sau khi các kiểm tra hoàn tất.

Chế độ này bắt buộc phải có `--data-root`.

Ví dụ không hợp lệ:

```powershell
python -m scripts.check_dataset --verify-hashes
```

Chương trình phải từ chối với thông báo:

`ValueError: --verify-hashes requires --data-root`

**Lưu ý:** Chế độ này đối chiếu tất cả ảnh được liệt kê trong manifest, không tự động chứng minh rằng thư mục giải nén không chứa file ngoài manifest.

### 6.5. Trực quan hóa Dataset

```powershell
python -m scripts.visualize_dataset --data-root data/raw
```

Script tạo các hình:

- `figures/class_distribution.png`
- `figures/augmentation_examples.png`

Các hình sử dụng dữ liệu thực tế khi script được chạy với Dataset đã giải nén.

Chỉ đưa hình vào báo cáo sau khi đã tạo và kiểm tra đầu ra thực tế.

---

## 7. Bằng chứng kiểm thử

### 7.1. Kiểm thử NV1

Môi trường kiểm thử được báo cáo:

- Windows.
- Python 3.14.2.
- PyTorch 2.14.1.
- Torchvision 0.29.1.
- pytest 9.2.0.
- Môi trường Python riêng: `.venv`.

Lệnh kiểm thử:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Kết quả chạy ngày 28/10/2026:

`7 passed in 2.64s`

Ba bài test Dataset ban đầu:

1. `test_manifest_integrity`
2. `test_mapping_and_val_transform`
3. `test_loaders_with_synthetic_files`

Bốn bài test SHA-256 bổ sung:

1. `test_sha256_matches_actual_file`
2. `test_sha256_detects_modified_file`
3. `test_verify_hashes_requires_data_root`
4. `test_verify_hashes_rejects_mismatch`

Các trường hợp được xác nhận:

- Hash của file tạm được tính đúng.
- Hash thay đổi khi nội dung file bị sửa.
- Chế độ `--verify-hashes` từ chối chạy nếu không có `--data-root`.
- Chương trình phát hiện giá trị hash không khớp trong trường hợp mô phỏng.

**Giới hạn kiểm thử NV1:**

Bộ test sử dụng file tạm và mô phỏng cho các trường hợp cần thiết. Kết quả này không thay thế cho việc chạy chế độ `--verify-hashes` trên toàn bộ 6.502 ảnh thật.

### 7.2. Kiểm thử độc lập của NV2

Theo log do NV2 đăng tại PR #8:

- Tải và giải nén bộ ZIP chứa 6.502 ảnh.
- Kiểm tra đường dẫn ảnh thành công.
- Kiểm tra tính toàn vẹn ZIP và các file ảnh.
- Load batch CNN thực tế.
- Xác nhận thứ tự nhãn đúng theo class mapping.
- Batch Train có shape `[32, 3, 224, 224]`.
- Validation transform có kết quả tất định.
- CNN chưa huấn luyện thực hiện forward pass thành công.

NV2 cũng báo cáo kết quả tính lại SHA-256 trực tiếp từ từng ảnh:

| Hạng mục | Kết quả |
|---|---:|
| Số dòng manifest | 6.502 |
| Hash khớp | 6.502 |
| Hash không khớp | 0 |
| File thiếu | 0 |
| File thừa trong thư mục ảnh đã kiểm tra | 0 |

Đây là bằng chứng kiểm thử độc lập của NV2, được ghi trong bình luận PR #8 ngày 23/10/2026.

Kết quả của NV2 không thay thế cho kiểm thử độc lập của NV3.

### 7.3. Kiểm thử NV3

NV3 đã kiểm tra cấu trúc tĩnh của manifest, split CSV và giao diện dữ liệu ResNet18.

Tuy nhiên, theo review hiện có, NV3 chưa cung cấp bằng chứng hoàn tất việc load batch ResNet18 trên bộ ảnh thật.

Các bước còn cần thực hiện:

```powershell
python -m scripts.check_dataset --data-root data/raw
```

Sau đó sử dụng:

```python
from src.data.dataset import create_dataloaders

loaders, datasets, mapping = create_dataloaders(
    data_root="data/raw",
    model_name="resnet18",
    batch_size=8
)
```

NV3 cần xác nhận:

- Có thể đọc ảnh thật từ Dataset.
- Tensor batch có shape `[8, 3, 224, 224]`.
- Class mapping sử dụng đúng các ID 0–3.
- Validation transform tất định.
- Adapter ResNet18 tương thích với DataLoader chung.
- Ghi lại log kiểm thử trong PR #8.

**Chưa đánh dấu nghiệm thu NV3 khi chưa có log.**

---

## 8. Checklist nghiệm thu Gate 1

### 8.1. Dữ liệu và mã nguồn

- [x] Có manifest cho 6.502 ảnh.
- [x] Có bảng thống kê số lượng theo từng lớp.
- [x] Có Train/Validation/Test CSV cố định.
- [x] Có class mapping gồm 4 lớp.
- [x] Có kiểm toán ảnh trùng SHA-256.
- [x] Có kiểm toán group overlap theo phương pháp đã mô tả.
- [x] Có Dataset và DataLoader chung.
- [x] Có preprocessing cho CNN và ResNet18.
- [x] Có script chuẩn bị và kiểm tra Dataset.
- [x] Có tài liệu hướng dẫn sử dụng.

### 8.2. Kiểm thử

- [x] Unit tests Dataset ban đầu PASS.
- [x] Bổ sung tùy chọn `--verify-hashes`.
- [x] Có unit tests cho SHA-256.
- [x] Tổng cộng 7/7 bài test NV1 PASS.
- [x] NV2 đã kiểm tra bộ ảnh thật và xác nhận 6.502 hash khớp.
- [x] NV2 đã load batch CNN và xác nhận mapping/shape.
- [ ] NV1 chạy trực tiếp script mới với `--verify-hashes` trên toàn bộ ảnh thật.
- [ ] NV3 load batch ResNet18 thực tế và cung cấp log.
- [ ] NV4 xác nhận giao diện transform tương thích Demo.
- [ ] Nhóm xác nhận chính sách preprocessing chính thức trước khi huấn luyện.

### 8.3. Review và tích hợp

- [x] Có PR #8 cho Dataset & Preprocessing.
- [x] Loại thay đổi `configs/cnn.yaml` khỏi phạm vi PR #8.
- [x] Đã bổ sung cơ chế kiểm tra SHA-256 và unit tests tại môi trường phát triển.
- [ ] Push toàn bộ thay đổi SHA-256, tests và tài liệu lên PR #8.
- [ ] Cung cấp hoặc liên kết đầy đủ log và hình trực quan hóa cần thiết.
- [ ] NV2 re-review và phê duyệt.
- [ ] NV3 re-review và phê duyệt.
- [ ] PR #8 được merge vào `main`.
- [ ] Issue #2 được xác nhận hoàn thành và đóng.

---

## 9. Trạng thái bàn giao

### Đã hoàn thành

- Xây dựng và cố định Dataset 6.502 ảnh.
- Hoàn thiện manifest, split CSV và class mapping.
- Triển khai DataLoader và preprocessing dùng chung.
- Kiểm tra tính nhất quán dữ liệu.
- Bổ sung cơ chế kiểm tra SHA-256 ảnh thật.
- Bổ sung các bài test cho chức năng mới.
- Kiểm thử NV1 đạt 7/7 bài test PASS.
- Nhận được bằng chứng NV2 đã kiểm tra dữ liệu và load batch CNN thực tế.
- Đã xử lý xung đột phạm vi sở hữu `configs/cnn.yaml`.

### Chưa hoàn thành

- Chưa có log chạy script `--verify-hashes` mới trên toàn bộ ảnh thật tại môi trường NV1.
- Chưa có log xác nhận Gate 1 độc lập từ NV3 trên ảnh thật.
- Chưa có phê duyệt cuối cùng từ các reviewer.
- PR #8 chưa được xác nhận merge.
- Issue #2 chưa được xác nhận đóng.

### Kết luận

Phần Dataset & Preprocessing đã có mã nguồn, dữ liệu cố định, tài liệu và kết quả kiểm thử hỗ trợ tích hợp CNN/ResNet18.

NV2 đã xác minh dữ liệu và khả năng đọc batch CNN trên ảnh thật.

Tuy nhiên, **Gate 1 chưa được coi là nghiệm thu hoàn tất** cho đến khi các điều kiện review còn lại được giải quyết và PR #8 được phê duyệt, merge.

Không sử dụng kết quả forward pass của mô hình chưa huấn luyện làm bằng chứng Accuracy hoặc chất lượng phân loại bệnh.

---

## 10. Liên kết tham khảo nội bộ

- [Issue #2 – Dataset & Preprocessing](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/issues/2)
- [PR #8 – NV1 Dataset](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/8)
- [PR #9 – NV2 CNN](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/9)
- [PR #7 – NV3 ResNet18](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/7)
- [PR #1 – NV4 Demo](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/1)

**Nguyên tắc:** Chỉ sử dụng kết quả thực nghiệm có log và nguồn kiểm chứng. Phân biệt rõ kết quả đã chạy, kết quả do thành viên khác báo cáo, mô phỏng kiểm thử và công việc chưa hoàn thành.
