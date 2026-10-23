# ResNet18 & Transfer Learning

Phụ trách: NV3 - Trần Văn Chính. Trạng thái: Chưa làm.

## 1. Cách chạy

- Huấn luyện (feature extraction + fine-tuning): (chưa điền)
- Đánh giá trên Test: (chưa điền)
- Dự đoán một ảnh: (chưa điền)

## 2. Mô hình

- Pretrained weights (tên/phiên bản torchvision): (chưa điền)
- Thay classification head: (chưa điền)
- Tổng số tham số / số tham số huấn luyện được: (chưa điền)

## 3. Tiền xử lý

(chưa điền) – resize/normalize theo yêu cầu pretrained weights, khác gì so với CNN.

## 4. Bước A – Feature extraction

| Tham số       | Giá trị     |
|---------------|-------------|
| Optimizer     | (chưa điền) |
| Learning rate | (chưa điền) |
| Epochs        | (chưa điền) |
| Batch size    | (chưa điền) |

## 5. Bước B – Fine-tuning

| Tham số                 | Giá trị     |
|-------------------------|-------------|
| Các tầng được mở băng   | (chưa điền) |
| LR backbone / LR head   | (chưa điền) |
| Optimizer               | (chưa điền) |
| Epochs                  | (chưa điền) |
| Early stopping          | (chưa điền) |

## 6. Môi trường

- Phần cứng (CPU/GPU): (chưa điền)
- Python / PyTorch / torchvision: (chưa điền)
- Seed: (chưa điền)

## 7. So sánh Feature Extraction và Fine-tuning (trên Validation)

Kết quả lưu ở `results/resnet18_feature_vs_finetune.csv` (chưa có).

| Cấu hình            | Val loss    | Val accuracy | Val macro F1 |
|---------------------|-------------|--------------|--------------|
| Feature extraction  | (chưa điền) | (chưa điền)  | (chưa điền)  |
| Fine-tuning         | (chưa điền) | (chưa điền)  | (chưa điền)  |

- Cấu hình cuối được chọn và lý do: (chưa điền)

## 8. Kết quả trên Test (sau khi chốt cấu hình)

- Biểu đồ: `figures/resnet18_train_curves.png` (chưa có)
- Confusion matrix: (chưa có)

| Lớp            | Precision   | Recall      | F1          |
|----------------|-------------|-------------|-------------|
| Healthy        | (chưa điền) | (chưa điền) | (chưa điền) |
| Early Blight   | (chưa điền) | (chưa điền) | (chưa điền) |
| Late Blight    | (chưa điền) | (chưa điền) | (chưa điền) |
| Bacterial Spot | (chưa điền) | (chưa điền) | (chưa điền) |
| Macro avg      | (chưa điền) | (chưa điền) | (chưa điền) |

- Accuracy (Test): (chưa điền)

## 9. Hạn chế

(chưa điền)

## Điều kiện nghiệm thu (mục V.4)

- [ ] Có mô hình pretrained ResNet18 với 4 đầu ra.
- [ ] Có bằng chứng chạy feature extraction và fine-tuning.
- [ ] Không chia lại dataset hoặc dùng Test để chọn cấu hình.
- [ ] Checkpoint tải lại dự đoán đúng mapping, pipeline và shape.
- [ ] Có log, metrics, biểu đồ và đánh giá hạn chế.
