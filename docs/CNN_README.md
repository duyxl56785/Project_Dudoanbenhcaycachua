# CNN tự xây dựng

Phụ trách: NV2 - Nguyễn Văn Duy. Trạng thái: Chưa làm.

## 1. Cách chạy

- Huấn luyện: (chưa điền)
- Đánh giá trên Test: (chưa điền)
- Dự đoán một ảnh: (chưa điền)

## 2. Kiến trúc

(chưa điền) – mô tả từng khối/lớp, kích thước tensor qua từng tầng.

- Tổng số tham số: (chưa điền)

## 3. Cấu hình huấn luyện

Giá trị đầy đủ trong `configs/cnn.yaml`.

| Tham số           | Giá trị     | Lý do       |
|-------------------|-------------|-------------|
| Loss              | (chưa điền) | (chưa điền) |
| Optimizer         | (chưa điền) | (chưa điền) |
| Learning rate     | (chưa điền) | (chưa điền) |
| LR scheduler      | (chưa điền) | (chưa điền) |
| Batch size        | (chưa điền) | (chưa điền) |
| Epochs            | (chưa điền) | (chưa điền) |
| Early stopping    | (chưa điền) | (chưa điền) |
| Checkpoint policy | (chưa điền) | (chưa điền) |
| Seed              | (chưa điền) |             |

## 4. Môi trường

- Phần cứng (CPU/GPU): (chưa điền)
- Python / PyTorch / torchvision: (chưa điền)

## 5. Thử nghiệm có kiểm soát (chỉ chọn bằng Validation)

(chưa điền)

## 6. Kết quả

Chỉ điền số liệu thực tế, truy vết được đến log và checkpoint.

- Biểu đồ loss/accuracy: `figures/cnn_train_curves.png` (chưa có)
- Metrics: (chưa có)
- Confusion matrix: (chưa có)

| Lớp            | Precision   | Recall      | F1          |
|----------------|-------------|-------------|-------------|
| Healthy        | (chưa điền) | (chưa điền) | (chưa điền) |
| Early Blight   | (chưa điền) | (chưa điền) | (chưa điền) |
| Late Blight    | (chưa điền) | (chưa điền) | (chưa điền) |
| Bacterial Spot | (chưa điền) | (chưa điền) | (chưa điền) |
| Macro avg      | (chưa điền) | (chưa điền) | (chưa điền) |

- Accuracy (Test): (chưa điền)

## 7. Phân tích

(chưa điền) – điểm mạnh, điểm yếu, các nhầm lẫn, dấu hiệu overfitting.

## Điều kiện nghiệm thu (mục IV.4)

- [ ] CNN huấn luyện TỪ ĐẦU, đầu ra đúng 4 lớp.
- [ ] Cùng split và mapping với NV1, không dùng Test để chọn model.
- [ ] Checkpoint chạy inference được trên máy khác theo README này.
- [ ] Có log/biểu đồ/metrics thực tế.
- [ ] Chỉ rõ điểm mạnh, điểm yếu, nhầm lẫn và dấu hiệu overfitting.
