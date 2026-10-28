# Quy trình đánh giá và bàn giao NV5

`scripts/evaluate_models.py` nhận hai file prediction đã chạy trên cùng một
Test split. Script dừng với lỗi nếu số mẫu, định danh ảnh hoặc ground truth
khác nhau giữa CNN và ResNet18. Nhãn có thể là số nguyên hoặc tên lớp; mapping
duy nhất được đọc từ `configs/class_mapping.json`.

## Đầu vào bắt buộc

Các script của NV2/NV3 xuất tối thiểu các cột sau:

| Vai trò | Tên cột được chấp nhận |
|---|---|
| Định danh mẫu | `path`, `image_path`, `sample_id`, `image_id`, `id`, `filename` |
| Nhãn thật | `true`, `true_label`, `label`, `target`, `y_true` |
| Dự đoán | `pred`, `predicted`, `predicted_label`, `prediction`, `y_pred` |

Nếu không có cột định danh, script dùng số thứ tự dòng. Cách này chỉ an toàn
khi hai file được xuất từ cùng một DataLoader không xáo trộn; nên luôn bàn giao
`path` để kiểm tra leakage và thứ tự mẫu.

Metadata metrics cạnh file prediction (`cnn_metrics.json` hoặc
`resnet18_metrics.json`) được tự đọc nếu có. Các trường như `checkpoint`,
`device`, `total_parameters`, `train_seconds` và `inference_seconds` sẽ được
đưa vào bảng nếu NV2/NV3 đã ghi chúng. Trường còn thiếu để trống, không được
điền số minh họa.

## Chạy

```bash
python -m scripts.evaluate_models \
  --cnn-predictions results/cnn/cnn_test_predictions.csv \
  --resnet18-predictions results/resnet18/resnet18_test_predictions.csv \
  --output-dir results
```

Đầu ra gồm:

- `results/comparison.csv`: Accuracy, macro Precision/Recall/F1, F1 từng lớp,
  số mẫu và các metadata đã bàn giao.
- `results/analysis.md`: nguồn file, kết quả, cặp lớp nhầm nhiều và giới hạn
  diễn giải.
- `results/cnn_confusion_matrix.png` và
  `results/resnet18_confusion_matrix.png`.

Đây là bước đánh giá sau khi checkpoint đã được chọn bằng Validation. Không
dùng `comparison.csv` để chọn lại mô hình trên Test.

## Checklist bàn giao cho NV5

- Gửi commit/nhánh, lệnh chạy, phiên bản Python/PyTorch/torchvision và thiết bị.
- Gửi checkpoint đã chọn, metadata checkpoint, prediction CSV và metrics JSON.
- Xác nhận manifest Test, class mapping, seed và số mẫu.
- Ghi rõ số liệu chạy lại độc lập hay lấy từ log của tác giả.
- Gửi ví dụ ảnh đúng/sai bằng `path`; không chép ảnh hoặc số liệu không có nguồn.
