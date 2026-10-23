"""Đánh giá thống nhất và so sánh CNN với ResNet18 trên Test chung.

Phụ trách: NV5 - Phạm Thế Cường.

Việc cần làm (mục II.D, VII tài liệu bàn giao):
- Kiểm tra model names, seed, nhãn, split, số ảnh, các cột metrics.
- Lập bảng so sánh: Accuracy, macro Precision/Recall/F1, per-class F1,
  số tham số, thời gian huấn luyện/inference (ghi rõ CPU/GPU, batch size,
  số ảnh).
- Phân tích cặp lớp dễ nhầm; ví dụ ảnh Test dự đoán đúng/sai.
- Ghi rõ phần nào đã chạy lại, phần nào dựa trên log của tác giả.
- Xuất results/comparison.csv và results/analysis.md.
"""
