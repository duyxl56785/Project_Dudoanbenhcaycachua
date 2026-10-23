"""Dự đoán một ảnh bằng checkpoint ResNet18.

Phụ trách: NV3 - Trần Văn Chính.

Việc cần làm (mục II.C, V tài liệu bàn giao):
- Nhận đường dẫn một ảnh, xử lý đúng pipeline của ResNet18.
- In top-1 và xác suất softmax 4 lớp theo configs/class_mapping.json.
- Kiểm tra checkpoint tải lại dự đoán đúng mapping, pipeline và shape.
"""
