"""Pipeline tiền xử lý và augmentation dùng chung cho CNN và ResNet18.

Phụ trách: NV1 - Lê Xuân Bảo.

Việc cần làm (mục III.2 tài liệu bàn giao):
- Resize, convert RGB, chuyển tensor, normalization.
- Tách pipeline riêng cho CNN/ResNet18 nếu khác nhau; ghi rõ giá trị dùng
  cho từng mô hình.
- Augmentation CHỈ cho Train (tài liệu gợi ý: xoay nhẹ, lật nếu có ý nghĩa,
  color jitter vừa phải, crop hợp lý - chưa chốt). Tránh làm mất tổn thương
  hoặc đổi bản chất nhãn.
- Validation/Test: tiền xử lý tất định, không random flip/rotate/crop.
- Bàn giao transforms cho NV2, NV3 và NV4.
"""
