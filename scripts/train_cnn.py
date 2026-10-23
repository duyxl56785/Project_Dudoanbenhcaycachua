"""Huấn luyện CNN từ đầu.

Phụ trách: NV2 - Nguyễn Văn Duy.

Việc cần làm (mục IV tài liệu bàn giao):
- Đọc configs/cnn.yaml; dùng split và class_mapping chung của NV1.
- Đặt seed; hàm train/validate; ghi train/val loss và accuracy theo epoch.
- Lưu checkpoint tốt nhất theo Validation vào checkpoints/cnn_best.pt;
  cân nhắc early stopping.
- Thử nghiệm có kiểm soát 1-2 yếu tố, CHỈ chọn bằng Validation.
- Xuất figures/cnn_train_curves.png; log seed, phiên bản thư viện,
  learning rate, batch size, epochs, optimizer, checkpoint policy.
"""
