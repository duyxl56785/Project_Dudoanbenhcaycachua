"""API suy luận dùng chung cho demo và script.

Phụ trách: NV4 - Nguyễn Minh Duy.

Việc cần làm (mục II.C, VI tài liệu bàn giao):
- Load checkpoint CNN và ResNet18 theo đúng định nghĩa mô hình của NV2/NV3;
  model.eval() và torch.no_grad()/inference_mode().
- Đầu vào: một ảnh RGB; áp dụng đúng preprocessing riêng của từng mô hình.
- Đầu ra: softmax 4 xác suất theo configs/class_mapping.json; không tự sắp
  xếp lại lớp. Xác suất này không đồng nghĩa chẩn đoán chắc chắn.
- Báo lỗi rõ khi thiếu checkpoint/ảnh lỗi, không crash ứng dụng.
- Kiểm thử: cùng một ảnh cho cùng kết quả với scripts/infer_*.py.
"""
