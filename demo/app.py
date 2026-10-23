"""Ứng dụng demo Streamlit: tải ảnh lá cà chua và dự đoán tình trạng lá.

Phụ trách: NV4 - Nguyễn Minh Duy.

Việc cần làm (mục VI tài liệu bàn giao):
- Giao diện: giới thiệu, upload JPG/PNG, preview ảnh, chọn một mô hình
  hoặc so sánh cả hai.
- Gọi src/inference.py; hiển thị top-1, xác suất 4 lớp, thời gian suy luận
  nếu đo đúng.
- MOCK (nếu dùng để thử giao diện) phải ghi nhãn rõ và gỡ bỏ trước nghiệm thu.
- Xử lý upload sai định dạng, ảnh lỗi, file quá lớn, thiếu checkpoint,
  lỗi phụ thuộc mà không crash ứng dụng.
- Thông báo hạn chế: chỉ phân loại 4 lớp đã học, không thay thế chuyên gia
  nông nghiệp.
"""
