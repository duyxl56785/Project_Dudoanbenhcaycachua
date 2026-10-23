# Demo Streamlit & tích hợp mô hình

Phụ trách: NV4 - Nguyễn Minh Duy. Trạng thái: Chưa làm.

## 1. Cài đặt và chạy

(chưa điền) – lệnh cài đặt trên môi trường sạch và lệnh chạy `demo/app.py`.

## 2. Checkpoint cần có

| Mô hình  | File checkpoint                | Config                  | Đã tích hợp |
|----------|--------------------------------|-------------------------|-------------|
| CNN      | `checkpoints/cnn_best.pt`      | `configs/cnn.yaml`      | Chưa        |
| ResNet18 | `checkpoints/resnet18_best.pt` | `configs/resnet18.yaml` | Chưa        |

Vị trí tải checkpoint: xem [checkpoints/README.md](../checkpoints/README.md).

## 3. Chức năng

(chưa điền) – upload JPG/PNG, preview, chọn mô hình hoặc so sánh cả hai, hiển thị top-1 và xác suất 4 lớp, thời gian suy luận.

## 4. Xử lý lỗi

(chưa điền) – sai định dạng, ảnh lỗi, file quá lớn, thiếu checkpoint, lỗi phụ thuộc.

## 5. Kiểm thử nhất quán

(chưa điền) – cùng ảnh cho cùng kết quả giữa `scripts/infer_*.py` và giao diện.

## 6. Ảnh chụp / video demo

Lưu ở `demo/screenshots/`. Đây là minh họa giao diện, không phải kết quả thực nghiệm.

(chưa điền)

## 7. Thông báo hạn chế hiển thị trên app

Mô hình chỉ phân loại 4 lớp trong tập đã học; kết quả là xác suất của mô hình, không phải chẩn đoán chắc chắn và không thay thế chuyên gia nông nghiệp.

## Điều kiện nghiệm thu (mục VI.4)

- [ ] Chạy app trên môi trường sạch theo README này.
- [ ] Load được CẢ HAI checkpoint thật, không hardcode nhãn/xác suất, đã gỡ mock.
- [ ] Đúng mapping, đúng normalization/resize, đúng kết quả kiểm thử.
- [ ] Có xử lý lỗi và thông báo hạn chế rõ ràng.
