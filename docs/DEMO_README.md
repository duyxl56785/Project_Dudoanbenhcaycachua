# Demo Streamlit – NV4 Nguyễn Minh Duy

## Trạng thái ngày 28/10/2026

Giao diện và API suy luận đã triển khai trên `feature/nv4-demo`; **Gate 4 chưa nghiệm thu**. PR [NV1 #8](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/8) đã merge vào `main`. Dataset thật và checkpoint CNN của PR [NV2 #9](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/9) đã được NV4 kiểm tra trong một bản tích hợp tạm trên CPU; PR #9 vẫn mở, chưa có review và chưa merge. PR [NV3 #7](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/7) đã được NV2 approve, nhưng chưa merge và chưa có checkpoint ResNet18 đã fine-tune. Vì vậy nhánh NV4 chưa chứa kiến trúc/checkpoint thật của hai mô hình và chế độ so sánh hai mô hình chưa thể nghiệm thu.

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

Upload JPEG/PNG tối đa **10 MiB**, xem trước ảnh, chọn CNN/ResNet18/so sánh cả hai, chọn thiết bị Tự động/CPU/CUDA rồi bấm **Dự đoán**. App hiển thị top-1, bảng và biểu đồ softmax bốn lớp theo thứ tự `configs/class_mapping.json`. Thời gian hiển thị chỉ đo forward pass, loại trừ tải mô hình, đọc ảnh và tiền xử lý; CUDA được đồng bộ trước và sau. Cache chỉ dùng lại mô hình khi file checkpoint không đổi. Softmax **không phải độ chắc chắn đã hiệu chỉnh**.

App kiểm tra nội dung ảnh, ảnh hỏng, giới hạn dung lượng, config/metadata, checkpoint và đầu ra logits. Khi thiếu checkpoint, app báo trước khi dự đoán; lỗi một mô hình không ngăn hiển thị kết quả mô hình còn lại trong chế độ so sánh. Chỉ đặt checkpoint do nhóm xác nhận tại `checkpoints/`. Checkpoint CNN hiện nằm trên nhánh NV2 và chưa được chép/commit vào NV4 trong khi PR #9 còn chờ review.

## Hợp đồng tích hợp

- NV1: `src.data.transforms.build_val_transform(model_name, config)` nhận `cnn` hoặc `resnet18`, trả callable ảnh PIL RGB → tensor CHW bằng pipeline Val/Test tất định. Config chính thức phải điền `image_size`, `normalize_mean`, `normalize_std`; ResNet18 cần thêm `resize_size`. Không dùng augmentation Train.
- NV2/NV3: `src.models.<name>.build_model(num_classes=4)` tạo đúng kiến trúc **không tải pretrained weights trong inference**. NV4 vẫn hỗ trợ `build_resnet18(num_classes=4, pretrained=False)` để tương thích phiên bản cũ của PR #7. Checkpoint `checkpoints/cnn_best.pt` / `checkpoints/resnet18_best.pt` phải là plain tensor `state_dict` load bằng `torch.load(..., weights_only=True)` và `load_state_dict(strict=True)`. Metadata/config lưu riêng; `<model>_best.meta.json` là bắt buộc để app kiểm nhãn và preprocessing khớp. Thứ tự logits luôn 0..3 theo mapping chung.
- Chọn checkpoint bằng `val_macro_f1` cao nhất, hòa thì `val_loss` thấp hơn; không dùng Test để chọn. Cần log Validation/commit/hash checkpoint để chứng minh, không suy ra từ tên file.

API: `predict_bytes(data, model_name, root=ROOT, device="cpu")` hoặc `predict_image(image, model_name, root=ROOT, device="cpu")`. `device="auto"` chọn CUDA khi khả dụng, nếu không thì CPU. `predict(...)` nhận model/transform để unit test; mô hình giả trong unit test không chứng minh tích hợp thật.

## Kiểm thử và bàn giao cho NV5

Đã chạy trong `.venv`: `python -m pytest tests -q` → **17 passed, 6 subtests passed**. `python -m compileall -q src demo tests` và `git diff --check` đạt. Streamlit khởi động với `--server.headless true --server.port 8506 --server.address 127.0.0.1`, health endpoint trả `ok`, sau đó server đã dừng. Health endpoint chỉ chứng minh server khởi động, không kiểm chứng thao tác upload trong trình duyệt.

Kiểm thử CNN thật dùng checkpoint từ head PR #9 trong bản sao tạm, không sửa nhánh NV2 và không đưa checkpoint vào commit NV4. Checkpoint có SHA-256 `d33c95bbe15e82fa83f1fd954136ebfff6e1cbba1a86fe40af473a377390de1`, metadata epoch 24, Validation macro-F1 0,976836. Trên ảnh Test nhãn Bacterial Spot `00416648-...3110.JPG` (SHA-256 `3320c467...db06`), `demo.check_consistency` xác nhận top-1 và cả bốn softmax của script NV2/API NV4 khớp tuyệt đối (`max |delta| = 0`) trên CPU. Đánh giá lại toàn bộ 1.356 ảnh Test trên CPU cho Accuracy **0,969764** và macro-F1 **0,965034**, khớp file kết quả NV2. Đây là kiểm chứng tái lập kỹ thuật của checkpoint đã chốt bằng Validation, không phải lựa chọn mô hình bằng Test.

ResNet18 từ head PR #7 đã nhận batch ảnh thật `[1,3,224,224]` và trả logits hữu hạn `[1,4]` với `pretrained=False`; đây chỉ là kiểm thử kiến trúc khởi tạo ngẫu nhiên. Chưa có checkpoint ResNet18 thật nên chưa chạy suy luận/đánh giá ResNet18 và chưa chạy `demo.check_consistency --model both`.

NV5 có thể dùng phần mô tả chức năng, cách chạy và giới hạn ở đây cho báo cáo. Chỉ bổ sung ảnh chụp vào `demo/screenshots/` sau khi chạy demo thật, ghi rõ đó là minh họa giao diện, không phải kết quả thực nghiệm. PlantVillage chủ yếu là ảnh trong điều kiện kiểm soát; ảnh cây/bệnh khác vẫn bị ép vào một trong bốn lớp. Không dùng demo để khẳng định chẩn đoán chắc chắn hay phát hiện sớm.

## Nghiệm thu VI.4

- [x] App khởi động trong `.venv` dựng từ `requirements.txt`; health endpoint trả `ok`. Tương tác trình duyệt chưa được tự động hóa.
- [ ] Load cả hai checkpoint thật: CNN đã đạt trong bản tích hợp tạm; ResNet18 chưa có checkpoint.
- [ ] Mapping, preprocessing và kết quả script khớp: CNN đã đối chiếu thật; ResNet18 còn chờ checkpoint.
- [x] UI xử lý ảnh lỗi, thiếu checkpoint và hiển thị hạn chế; mã và test độc lập đã có, cần xác minh lại trong môi trường đầy đủ.
