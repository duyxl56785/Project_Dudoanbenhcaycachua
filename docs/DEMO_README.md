# Demo Streamlit – NV4 Nguyễn Minh Duy

## Trạng thái ngày 28/10/2026

Giao diện và API suy luận đã triển khai trên `feature/nv4-demo`; **Gate 4 chưa nghiệm thu**. PR [NV1 #8](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/8) đã merge vào `main`. Dataset thật và checkpoint CNN của PR [NV2 #9](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/9) đã được NV4 kiểm tra trong một bản tích hợp tạm trên CPU; PR #9 vẫn mở, chưa có review và chưa merge. PR [NV3 #7](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/7) đã được NV2 approve, nhưng chưa merge và chưa có checkpoint ResNet18 đã fine-tune. Vì vậy nhánh NV4 chưa chứa kiến trúc/checkpoint thật của hai mô hình và chế độ so sánh hai mô hình chưa thể nghiệm thu.

## Giới thiệu và luồng hoạt động

Demo nhận một ảnh lá cà chua, xác minh nội dung JPEG/PNG, chuyển ảnh về RGB, áp dụng đúng transform Val/Test của mô hình, chạy forward pass trong `torch.inference_mode()`, rồi hiển thị nhãn top-1, bốn xác suất softmax và biểu đồ. Luồng chính:

`Upload → kiểm tra ảnh → RGB → transform NV1 → CNN/ResNet18 → logits 4 lớp → softmax → bảng/biểu đồ`

`demo/app.py` chỉ phụ trách giao diện. Mọi kiểm tra mapping, config, metadata, checkpoint và suy luận dùng chung `src/inference.py`, tránh để script và giao diện có hai pipeline khác nhau. Mô hình được cache theo tên, thiết bị và thời điểm sửa checkpoint; thay file trọng số sẽ làm cache hết hiệu lực.

## Bốn lớp đầu ra

Thứ tự cố định lấy từ `configs/class_mapping.json`, không sắp xếp theo alphabet:

| Chỉ số | Nhãn hiển thị |
|---:|---|
| 0 | Healthy |
| 1 | Early Blight |
| 2 | Late Blight |
| 3 | Bacterial Spot |

## Dataset đã xác minh trên máy NV4

- Nguồn nhóm: [Google Drive – ML_Tomato_Dataset/v1_manifests](https://drive.google.com/drive/folders/1gazFJNC1zdyAj8adjYeKTlOu1JJWIj7v), file `tomato_6502_images.zip`; nguồn gốc ảnh là [mohanty/PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage), cấu hình ảnh màu.
- ZIP được lưu cục bộ tại `data/tomato_6502_images.zip`, SHA-256 `303c77726cd56def12de4bec9a86517c6c65abd62b1dca8614f717fcfc227304`, kích thước 105.542.302 byte. ZIP và `data/raw/` đã được `.gitignore` loại khỏi commit.
- Ảnh giải nén nằm tại `data/raw/raw/color/`: **6.502 JPEG RGB 256×256**, gồm đúng bốn thư mục lớp. Train/Validation/Test giữ nguyên manifest: **4.115 / 1.031 / 1.356**.
- `python -m scripts.check_dataset --data-root data/raw --verify-hashes` đã đối chiếu thành công SHA-256 của cả 6.502 ảnh. Manifest có 6.502 path/hash duy nhất và không có path, SHA-256 hoặc `group_id` chồng lấn giữa các split theo quy tắc kiểm toán NV1. Pillow `verify()` đọc được toàn bộ ảnh.

Chuẩn bị lại dữ liệu trên máy khác:

```powershell
python -m scripts.prepare_dataset --archive data/tomato_6502_images.zip --data-root data/raw
python -m scripts.check_dataset --data-root data/raw --verify-hashes
```

## Cài đặt và chạy

`requirements.txt` đã liệt kê Streamlit, PyTorch, torchvision, Pillow và PyYAML. Môi trường `.venv` dựng từ file này đã được kiểm tra với Python 3.12.0, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, Streamlit 1.66.1, Pillow 13.0.0 và PyYAML 6.0.3. CNN đã được xác minh end-to-end trong bản tích hợp tạm; nhánh NV4 hiện vẫn thiếu mã/checkpoint đã merge. Các lệnh cần chạy từ thư mục gốc dự án:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run demo/app.py
python -m unittest discover -s tests -v
```

## Tiền xử lý và mô hình

Val/Test và demo dùng pipeline tất định từ `build_val_transform(model_name, config)`: chuyển RGB, resize/crop theo YAML, chuyển tensor và normalize bằng đúng `mean/std` của checkpoint; không dùng augmentation Train. CNN PR #9 dùng ảnh đầu vào 224×224 và ImageNet mean/std. ResNet18 phải dùng config/metadata chính thức của NV3; các giá trị còn `null` không được tự suy đoán.

- **CNN tự xây dựng:** `build_model(num_classes=4)`, checkpoint thật PR #9, đã strict-load và suy luận CPU thành công.
- **ResNet18:** source PR #7 có `build_model(num_classes=4)` nhưng chưa có checkpoint fine-tuned. Demo chỉ báo thiếu checkpoint, tuyệt đối không chạy trọng số ngẫu nhiên như kết quả thật.

## Chuẩn bị checkpoint

Mỗi mô hình cần ba thành phần đồng bộ: kiến trúc `build_model(4)`, YAML preprocessing và cặp `checkpoints/<model>_best.pt` + `<model>_best.meta.json`. File `.pt` phải là plain tensor `state_dict`; app dùng `torch.load(..., map_location=device, weights_only=True)` và `load_state_dict(strict=True)`. Metadata phải chứa đúng bốn nhãn theo mapping chung và preprocessing giống YAML. Chỉ sử dụng checkpoint được chọn bằng `val_macro_f1`, không chọn bằng Test.

## Sử dụng giao diện và đọc kết quả

Upload JPEG/PNG tối đa **10 MiB**, xem trước ảnh, chọn CNN/ResNet18/so sánh cả hai, chọn thiết bị Tự động/CPU/CUDA rồi bấm **Dự đoán**. App hiển thị top-1, bảng và biểu đồ softmax bốn lớp theo thứ tự `configs/class_mapping.json`. Thời gian hiển thị chỉ đo forward pass, loại trừ tải mô hình, đọc ảnh và tiền xử lý; CUDA được đồng bộ trước và sau. Cache chỉ dùng lại mô hình khi file checkpoint không đổi. Softmax **không phải độ chắc chắn đã hiệu chỉnh**.

App kiểm tra nội dung ảnh, ảnh hỏng, giới hạn dung lượng, config/metadata, checkpoint và đầu ra logits. Khi thiếu checkpoint, app báo trước khi dự đoán; lỗi một mô hình không ngăn hiển thị kết quả mô hình còn lại trong chế độ so sánh. Chỉ đặt checkpoint do nhóm xác nhận tại `checkpoints/`. Checkpoint CNN hiện nằm trên nhánh NV2 và chưa được chép/commit vào NV4 trong khi PR #9 còn chờ review.

Nếu chưa tải ảnh mà bấm **Dự đoán bệnh**, app yêu cầu tải JPG/PNG. Khi thay ảnh hoặc đổi mô hình, kết quả cũ được xóa sau khi lần rerun hoàn tất. Xác suất hiển thị là softmax tương đối giữa đúng bốn lớp đã học, không phải độ tin cậy đã hiệu chỉnh hay xác suất chẩn đoán ngoài thực địa.

## Hợp đồng tích hợp

- NV1: `src.data.transforms.build_val_transform(model_name, config)` nhận `cnn` hoặc `resnet18`, trả callable ảnh PIL RGB → tensor CHW bằng pipeline Val/Test tất định. Config chính thức phải điền `image_size`, `normalize_mean`, `normalize_std`; ResNet18 cần thêm `resize_size`. Không dùng augmentation Train.
- NV2/NV3: `src.models.<name>.build_model(num_classes=4)` tạo đúng kiến trúc **không tải pretrained weights trong inference**. NV4 vẫn hỗ trợ `build_resnet18(num_classes=4, pretrained=False)` để tương thích phiên bản cũ của PR #7. Checkpoint `checkpoints/cnn_best.pt` / `checkpoints/resnet18_best.pt` phải là plain tensor `state_dict` load bằng `torch.load(..., weights_only=True)` và `load_state_dict(strict=True)`. Metadata/config lưu riêng; `<model>_best.meta.json` là bắt buộc để app kiểm nhãn và preprocessing khớp. Thứ tự logits luôn 0..3 theo mapping chung.
- Chọn checkpoint bằng `val_macro_f1` cao nhất, hòa thì `val_loss` thấp hơn; không dùng Test để chọn. Cần log Validation/commit/hash checkpoint để chứng minh, không suy ra từ tên file.

API: `predict_bytes(data, model_name, root=ROOT, device="cpu")` hoặc `predict_image(image, model_name, root=ROOT, device="cpu")`. `device="auto"` chọn CUDA khi khả dụng, nếu không thì CPU. `predict(...)` nhận model/transform để unit test; mô hình giả trong unit test không chứng minh tích hợp thật.

## Kiểm thử và bàn giao cho NV5

Đã chạy trong `.venv`: `python -m pytest tests -q` → **29 passed, 19 subtests passed**. `python -m compileall -q src demo tests` và `git diff --check` đạt. Streamlit khởi động thành công ở `127.0.0.1`; sau khi kiểm thử server đã được dừng.

Kiểm thử CNN thật dùng checkpoint từ head PR #9 trong bản sao tạm, không sửa nhánh NV2 và không đưa checkpoint vào commit NV4. Checkpoint có SHA-256 `d33c95bbe15e82fa83f1fd954136ebfff6e1cbba1a86fe40af473a377390de1`, metadata epoch 24, Validation macro-F1 0,976836. Trên ảnh Test nhãn Bacterial Spot `00416648-...3110.JPG` (SHA-256 `3320c467...db06`), `demo.check_consistency` xác nhận top-1 và cả bốn softmax của script NV2/API NV4 khớp tuyệt đối (`max |delta| = 0`) trên CPU. Đánh giá lại toàn bộ 1.356 ảnh Test trên CPU cho Accuracy **0,969764** và macro-F1 **0,965034**, khớp file kết quả NV2. Đây là kiểm chứng tái lập kỹ thuật của checkpoint đã chốt bằng Validation, không phải lựa chọn mô hình bằng Test.

Kiểm thử trình duyệt tự động cũng dùng đúng bản tích hợp tạm nói trên và một ảnh Test thật. Ảnh xem trước hiển thị đúng; chọn CNN rồi bấm **Dự đoán** trả về `Bacterial Spot` cùng đủ bốn xác suất: Healthy 0,00%, Early Blight 2,99%, Late Blight 0,06%, Bacterial Spot 96,95%. File PNG giả nội dung ảnh hỏng được từ chối với thông báo không đọc được ảnh. Khi chọn ResNet18, app báo đúng `Thiếu checkpoint ... resnet18_best.pt`; thứ tự kiểm tra trong `src/inference.py` đã được sửa để lỗi thiếu checkpoint không bị che bởi config preprocessing chưa hoàn thiện, kèm regression test.

### Kết quả trình duyệt trên bốn lớp

Mỗi ảnh dưới đây là ảnh đầu tiên của lớp tương ứng trong split Test chính thức. Xác suất được ghi đúng như giao diện, kể cả trường hợp dự đoán sai.

| Ảnh Test | Nhãn thật | Nhãn CNN | Healthy | Early Blight | Late Blight | Bacterial Spot | Kết quả |
|---|---|---|---:|---:|---:|---:|---|
| `00bce074-...0223.JPG` (`41ae744b...a014`) | Healthy | Healthy | 99,97% | 0,00% | 0,03% | 0,00% | Đúng |
| `0012b9d2-...8389.JPG` (`29b79036...6123`) | Early Blight | Late Blight | 0,04% | 41,74% | 58,16% | 0,07% | Sai |
| `005e3b43-...5104.JPG` (`a9eebda1...49a3`) | Late Blight | Late Blight | 0,05% | 7,43% | 92,48% | 0,04% | Đúng |
| `00416648-...3110.JPG` (`3320c467...db06`) | Bacterial Spot | Bacterial Spot | 0,00% | 2,99% | 0,06% | 96,95% | Đúng |

Ảnh Early Blight bị nhầm thành Late Blight ở biên khá gần. Kết quả này phù hợp với báo cáo NV2: phần lớn lỗi Test là nhầm Early Blight ↔ Late Blight; đây không phải lỗi mapping hay giao diện. Đổi ảnh liên tiếp xóa kết quả cũ; chạy bốn ảnh liên tiếp cho cache `hits=3, misses=1`, chứng minh cùng checkpoint chỉ nạp một lần trong tiến trình API.

### Ma trận kiểm thử đầu vào và chuyển mô hình

| Trường hợp | Kết quả thực tế |
|---|---|
| Chưa upload rồi bấm dự đoán | Báo “Vui lòng tải ảnh JPG hoặc PNG…”; không traceback |
| TXT đổi tên `.jpg`, PNG hỏng | Bị Pillow từ chối; không có kết quả cũ |
| File rỗng | Báo ảnh trống hoặc vượt giới hạn |
| GIF | Bộ upload từ chối `image/gif`; không vào pipeline |
| File 10 MiB + 1 byte | Báo vượt giới hạn 10 MiB |
| Grayscale, RGBA | Chuyển RGB, preview và suy luận thành công |
| Ảnh 91×37 | Preview và suy luận thành công; transform resize về 224×224 |
| Chọn ResNet18 | Báo thiếu `resnet18_best.pt`, không dùng trọng số ngẫu nhiên |
| So sánh cả hai | CNN vẫn hiển thị kết quả; ResNet18 báo thiếu checkpoint riêng |
| Đổi CNN/ResNet18/so sánh và chạy lặp | Không crash; trạng thái kết quả được làm mới |

ResNet18 từ head PR #7 đã nhận batch ảnh thật `[1,3,224,224]` và trả logits hữu hạn `[1,4]` với `pretrained=False`; đây chỉ là kiểm thử kiến trúc khởi tạo ngẫu nhiên. Chưa có checkpoint ResNet18 thật nên chưa chạy suy luận/đánh giá ResNet18 và chưa chạy `demo.check_consistency --model both`.

### Ảnh bằng chứng demo thật

Các ảnh dưới đây được chụp trực tiếp từ phiên Streamlit kiểm thử, không phải ảnh dựng. Chúng là bằng chứng hành vi giao diện; số liệu đánh giá tập Test vẫn lấy từ phép kiểm thử tái lập nêu trên.

![CNN dự đoán ảnh lá thật](../demo/screenshots/cnn_prediction_real.jpg)

![Từ chối tệp ảnh hỏng](../demo/screenshots/invalid_image_error.jpg)

![ResNet18 báo thiếu checkpoint thật](../demo/screenshots/resnet18_missing_checkpoint.jpg)

NV5 có thể dùng phần mô tả chức năng, cách chạy, giới hạn và ảnh bằng chứng ở đây cho báo cáo. PlantVillage chủ yếu là ảnh trong điều kiện kiểm soát; ảnh cây/bệnh khác vẫn bị ép vào một trong bốn lớp. Không dùng demo để khẳng định chẩn đoán chắc chắn hay phát hiện sớm.

## Lỗi thường gặp và cách khắc phục

| Thông báo | Nguyên nhân | Cách xử lý |
|---|---|---|
| `Thiếu checkpoint ...` | Chưa đặt trọng số được xác nhận | Lấy checkpoint từ PR/commit chính thức; không tạo file giả |
| `Thiếu metadata ...` | Thiếu sidecar hoặc bàn giao chưa đủ | Đặt `.meta.json` cạnh checkpoint |
| `preprocessing chưa chốt` | YAML còn `null` | Người phụ trách mô hình chốt config và metadata cùng nhau |
| `thứ tự nhãn khác` / `preprocessing khác` | Checkpoint không đồng bộ cấu hình | Dùng đúng cặp checkpoint–metadata–YAML |
| `Không tải được checkpoint` | File hỏng, sai kiến trúc/số lớp/định dạng | Kiểm tra SHA-256, `build_model(4)` và plain `state_dict` |
| `CUDA chưa khả dụng` | Máy không có CUDA phù hợp | Chọn CPU hoặc Tự động |
| Không đọc được ảnh | Nội dung không phải JPEG/PNG hoặc file hỏng | Xuất lại ảnh hợp lệ, không chỉ đổi phần mở rộng |

## Giới hạn

- Chỉ phân loại bốn lớp đã học; không có lớp “khác/không biết”.
- PlantVillage chủ yếu chụp trong điều kiện kiểm soát, chưa chứng minh độ chính xác ngoài đồng ruộng.
- Softmax chưa hiệu chỉnh; không thay thế chuyên gia nông nghiệp.
- CNN đã được kiểm thử thật nhưng PR #9 chưa merge. ResNet18 chưa có checkpoint fine-tuned nên chưa thể so sánh hai mô hình hoặc nghiệm thu Gate 4.

## Nghiệm thu VI.4

- [x] App khởi động trong `.venv` dựng từ `requirements.txt`; trình duyệt đã tự động upload/xem trước ảnh thật, chọn CNN, dự đoán và hiển thị đủ bốn xác suất.
- [ ] Load cả hai checkpoint thật: CNN đã đạt trong bản tích hợp tạm; ResNet18 chưa có checkpoint.
- [ ] Mapping, preprocessing và kết quả script khớp: CNN đã đối chiếu thật; ResNet18 còn chờ checkpoint.
- [x] UI xử lý ảnh lỗi, thiếu checkpoint và hiển thị hạn chế; ảnh hỏng và ResNet18 thiếu checkpoint đã được xác minh trực tiếp trong trình duyệt.
