# Demo Streamlit – NV4 Nguyễn Minh Duy

## Trạng thái

Giao diện và API suy luận đã triển khai; **chưa nghiệm thu tích hợp thật**. Repository hiện có mapping bốn lớp, cấu hình YAML và khung mã. `src/data/transforms.py`, `src/models/cnn.py`, `src/models/resnet18.py` và `scripts/infer_*.py` vẫn là khung; các giá trị Val/Test trong YAML là `null`; chưa có checkpoint ở `checkpoints/`. Không có số liệu dự đoán thật hay ảnh chụp demo.

## Cài đặt, chạy

Đã kiểm tra cú pháp với Python 3.12.0; môi trường hiện có Pillow 12.1.1, PyYAML 6.0.3 và Streamlit 1.58.0. Chưa xác nhận cài đặt sạch của toàn bộ `requirements.txt` vì môi trường hiện thiếu PyTorch và không truy cập được mạng. Lệnh khởi chạy Streamlit đã được thử nhưng môi trường chặn socket cục bộ; chưa xác minh giao diện bằng trình duyệt.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run demo/app.py
python -m unittest discover -s tests -v
```

Upload JPEG/PNG tối đa **10 MiB**, chọn CNN, ResNet18 hoặc so sánh cả hai. App xem trước ảnh, rồi hiển thị top-1, softmax bốn lớp theo thứ tự `configs/class_mapping.json` và thời gian **chỉ của forward pass** (không gồm tải mô hình/ảnh/transform; CUDA được đồng bộ trước và sau). Softmax chưa được hiệu chỉnh. File sai nội dung, ảnh hỏng, quá lớn, thiếu cấu hình/checkpoint hoặc mô hình không tương thích sẽ báo lỗi tại giao diện.

## Hợp đồng bàn giao để tích hợp

- NV1: `src.data.transforms.build_val_transform(model_name, config)` trả callable biến ảnh PIL RGB thành tensor CHW bằng pipeline Val/Test tất định đúng cho từng mô hình. Ghi rõ resize, mean/std và điền các giá trị `null` tương ứng trong YAML; không dùng augmentation Train.
- NV2: `src.models.cnn.build_model(num_classes=4)` trả đúng kiến trúc đã train; `checkpoints/cnn_best.pt` là **plain tensor state_dict** từ mô hình đó. Bàn giao script `scripts/infer_cnn.py` chạy được, commit và nguồn checkpoint nhóm xác nhận.
- NV3: tương tự với `src.models.resnet18.build_model(num_classes=4)`, `checkpoints/resnet18_best.pt` và `scripts/infer_resnet18.py`; kiến trúc phải load checkpoint mà không tải pretrained weights từ mạng khi suy luận.
- Cả hai: thứ tự output logits chính xác 0..3 của `configs/class_mapping.json`; checkpoint được chọn bằng Validation. Nếu định dạng checkpoint thực tế khác plain state_dict, cần thống nhất và sửa loader nhỏ, có kiểm thử tương ứng. Chỉ đặt checkpoint nhận từ nguồn nhóm xác nhận; hiện **chưa có link xác minh**.

API `predict_bytes(data, model_name, root=ROOT, device="cpu")` dùng cho app và có thể dùng lại ở script. `predict(image, model, transform, names, device)` cho unit test với mô hình giả; kết quả unit test đó không chứng minh tích hợp mô hình thật. CPU là mặc định.

## Kiểm thử nhất quán cần thực hiện sau bàn giao

Chọn một ảnh JPEG/PNG cố định và ghi hash/đường dẫn, commit mã, checkpoint và hash, phiên bản thư viện, thiết bị. Chạy cùng ảnh qua `scripts/infer_cnn.py` và `scripts/infer_resnet18.py`, rồi qua `predict_bytes`; đối chiếu top-1 và cả bốn xác suất với sai số nhỏ (ví dụ `1e-5`). Lưu lệnh, đầu ra và kết quả vào PR. **Chưa thực hiện vì thiếu scripts và checkpoint thật.**

## Nghiệm thu VI.4

- [ ] App chạy trên môi trường sạch: chưa xác minh.
- [ ] Load cả hai checkpoint thật: chưa có checkpoint và kiến trúc.
- [ ] Mapping, normalization/resize và kết quả script khớp: mapping đã kiểm thử độc lập; transform và đối chiếu thật còn chờ.
- [x] UI có xử lý ảnh lỗi/thiếu đầu vào và thông báo hạn chế; kiểm thử mã độc lập, chưa chạy Streamlit trong môi trường này.

Ảnh chụp nếu có sau khi chạy demo thật đặt tại `demo/screenshots/` và ghi rõ chỉ minh họa giao diện, không phải kết quả thực nghiệm.
