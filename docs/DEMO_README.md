# Demo Streamlit – NV4 Nguyễn Minh Duy

## Trạng thái ngày 28/10/2026

Giao diện và API suy luận đã triển khai trên `feature/nv4-demo`; **Gate 4 chưa nghiệm thu**. PR [NV1 #8](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/8) đã merge vào `main` và cung cấp `build_val_transform` cùng manifest 6.502 ảnh. PR [NV2 #9](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/9) có CNN, `build_model`, cấu hình 224×224/ImageNet, script và định dạng checkpoint `state_dict` thuần + `.meta.json`. PR [NV3 #7](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/7) đã bổ sung `build_model`, checkpoint `state_dict` thuần + metadata và dùng transform chung của NV1. Hai PR mô hình chưa merge và chưa bàn giao checkpoint đã huấn luyện; không coi kiểm thử mô hình khởi tạo ngẫu nhiên là kết quả thực nghiệm.

## Cài đặt và chạy

`requirements.txt` đã liệt kê Streamlit, PyTorch, torchvision, Pillow và PyYAML. Môi trường `.venv` dựng từ file này đã được kiểm tra với Python 3.12.0, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, Streamlit 1.66.1, Pillow 13.0.0 và PyYAML 6.0.3. **Chưa xác minh dự đoán end-to-end** do chưa có checkpoint. Các lệnh cần chạy từ thư mục gốc dự án:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run demo/app.py
python -m unittest discover -s tests -v
```

Upload JPEG/PNG tối đa **10 MiB**, xem trước ảnh, chọn CNN/ResNet18/so sánh cả hai, chọn thiết bị Tự động/CPU/CUDA rồi bấm **Dự đoán**. App hiển thị top-1, bảng và biểu đồ softmax bốn lớp theo thứ tự `configs/class_mapping.json`. Thời gian hiển thị chỉ đo forward pass, loại trừ tải mô hình, đọc ảnh và tiền xử lý; CUDA được đồng bộ trước và sau. Cache chỉ dùng lại mô hình khi file checkpoint không đổi. Softmax **không phải độ chắc chắn đã hiệu chỉnh**.

App kiểm tra nội dung ảnh, ảnh hỏng, giới hạn dung lượng, config/metadata, checkpoint và đầu ra logits. Khi thiếu checkpoint, app báo trước khi dự đoán; lỗi một mô hình không ngăn hiển thị kết quả mô hình còn lại trong chế độ so sánh. Chỉ đặt checkpoint do nhóm xác nhận tại `checkpoints/`; chưa có link tải đã xác minh.

## Hợp đồng tích hợp

- NV1: `src.data.transforms.build_val_transform(model_name, config)` nhận `cnn` hoặc `resnet18`, trả callable ảnh PIL RGB → tensor CHW bằng pipeline Val/Test tất định. Config chính thức phải điền `image_size`, `normalize_mean`, `normalize_std`; ResNet18 cần thêm `resize_size`. Không dùng augmentation Train.
- NV2/NV3: `src.models.<name>.build_model(num_classes=4)` tạo đúng kiến trúc **không tải pretrained weights trong inference**. NV4 vẫn hỗ trợ `build_resnet18(num_classes=4, pretrained=False)` để tương thích phiên bản cũ của PR #7. Checkpoint `checkpoints/cnn_best.pt` / `checkpoints/resnet18_best.pt` phải là plain tensor `state_dict` load bằng `torch.load(..., weights_only=True)` và `load_state_dict(strict=True)`. Metadata/config lưu riêng; `<model>_best.meta.json` là bắt buộc để app kiểm nhãn và preprocessing khớp. Thứ tự logits luôn 0..3 theo mapping chung.
- Chọn checkpoint bằng `val_macro_f1` cao nhất, hòa thì `val_loss` thấp hơn; không dùng Test để chọn. Cần log Validation/commit/hash checkpoint để chứng minh, không suy ra từ tên file.

API: `predict_bytes(data, model_name, root=ROOT, device="cpu")` hoặc `predict_image(image, model_name, root=ROOT, device="cpu")`. `device="auto"` chọn CUDA khi khả dụng, nếu không thì CPU. `predict(...)` nhận model/transform để unit test; mô hình giả trong unit test không chứng minh tích hợp thật.

## Kiểm thử và bàn giao cho NV5

Đã chạy trong `.venv`: `python -m pytest tests -q` → **17 passed, 6 subtests passed**. `python -m compileall -q src demo tests` và `git diff --check` đạt. `streamlit.testing.v1.AppTest` không trả kết quả sau hơn một phút và đã được dừng; thay vào đó `python -m streamlit run demo/app.py --server.headless true --server.port 8504 --server.address 127.0.0.1` khởi động server, `GET http://127.0.0.1:8504/_stcore/health` trả `ok`, sau đó server đã dừng. Chưa có checkpoint thật để đối chiếu script và API.

Sau khi nhận hai checkpoint đã huấn luyện, chọn **cùng một ảnh**, lưu path và SHA-256 ảnh, path/SHA-256 checkpoint, commit của NV1/NV2/NV3/NV4, phiên bản Python/torch/torchvision/Streamlit và thiết bị. Chạy `python -m scripts.infer_cnn <ảnh>` và `python -m scripts.infer_resnet18 <ảnh>`, rồi `predict_bytes` cho mỗi mô hình. Đối chiếu top-1 và cả bốn softmax với sai số `1e-5`; ghi lệnh và đầu ra vào PR. **Chưa thực hiện do thiếu checkpoint thật và PR phụ thuộc chưa merge.**

Khi scripts và checkpoint đã sẵn sàng, chạy `python -m demo.check_consistency <ảnh> --model both`. Script dùng CPU cho API và vô hiệu CUDA ở tiến trình script, so sánh top-1 cùng bốn xác suất với ngưỡng `1e-5`, trả mã lỗi nếu khác. Bài unit test của bộ so sánh chỉ kiểm logic so sánh bằng số giả; **chưa phải phép đối chiếu mô hình thật**.

NV5 có thể dùng phần mô tả chức năng, cách chạy và giới hạn ở đây cho báo cáo. Chỉ bổ sung ảnh chụp vào `demo/screenshots/` sau khi chạy demo thật, ghi rõ đó là minh họa giao diện, không phải kết quả thực nghiệm. PlantVillage chủ yếu là ảnh trong điều kiện kiểm soát; ảnh cây/bệnh khác vẫn bị ép vào một trong bốn lớp. Không dùng demo để khẳng định chẩn đoán chắc chắn hay phát hiện sớm.

## Nghiệm thu VI.4

- [ ] App chạy trong môi trường sạch: chưa xác minh.
- [ ] Load cả hai checkpoint thật: chưa có checkpoint đã huấn luyện được bàn giao.
- [ ] Mapping, preprocessing và kết quả script khớp: mapping được kiểm thử độc lập; đối chiếu thật còn chờ.
- [x] UI xử lý ảnh lỗi, thiếu checkpoint và hiển thị hạn chế; mã và test độc lập đã có, cần xác minh lại trong môi trường đầy đủ.
