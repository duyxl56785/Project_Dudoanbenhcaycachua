# Quy trình đánh giá và bàn giao NV5

`scripts/evaluate_models.py` chỉ tổng hợp artifact đã được chủ mô hình bàn giao.
Script không huấn luyện, không chia lại dữ liệu và không chọn checkpoint. Khi
có hai prediction CSV, script dừng với lỗi nếu số mẫu, định danh ảnh hoặc ground
truth khác nhau giữa CNN và ResNet18. Nhãn có thể là số nguyên hoặc tên lớp;
mapping duy nhất được đọc từ `configs/class_mapping.json`.

## Đầu vào bắt buộc

Các script của NV2/NV3 xuất tối thiểu các cột sau:

| Vai trò | Tên cột được chấp nhận |
|---|---|
| Định danh mẫu | `path`, `image_path`, `sample_id`, `image_id`, `id`, `filename` |
| Nhãn thật | `true`, `true_label`, `label`, `target`, `y_true` |
| Dự đoán | `pred`, `predicted`, `predicted_label`, `prediction`, `y_pred` |

Nên có thêm cột `split` (hoặc `dataset_split`, `subset`) với giá trị `test`.
CSV phải dùng UTF-8/UTF-8 BOM, có header, không rỗng và không trùng sample ID.
Mẫu header được khuyến nghị:

```csv
path,true,pred,split
data/test/healthy_001.jpg,Healthy,Healthy,test
```

Nếu không có cột định danh, script dùng số thứ tự dòng. Cách này chỉ an toàn
khi hai file được xuất từ cùng một DataLoader không xáo trộn; nên luôn bàn giao
`path` để kiểm tra leakage và thứ tự mẫu.

Metadata metrics cạnh file prediction (`cnn_metrics.json` hoặc
`resnet18_metrics.json`) được tự đọc nếu có; cũng có thể chỉ định tường minh
bằng `--cnn-metrics` hay `--resnet18-metrics`. JSON cần có `n_samples`,
`accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `per_class` đủ bốn
lớp (gồm `precision`, `recall`, `f1`, `support`) và `confusion_matrix` 4×4 nếu
có ma trận. Script kiểm tra tổng `support` và tổng matrix khớp `n_samples`.

Các trường `checkpoint`, `checkpoint_epoch`, `checkpoint_val_macro_f1`,
`checkpoint_val_loss`, `device`, `seed`, `total_parameters`,
`trainable_parameters`, `train_seconds`, `inference_seconds`, phiên bản Python/
PyTorch/torchvision chỉ được đưa vào bảng nếu NV2/NV3 đã ghi. Trường còn thiếu
để trống, không được điền số minh họa.

Adapter NV5 cũng nhận JSON theo `classification_report` của scikit-learn do NV3
bàn giao: lấy `macro avg` và bốn lớp để chuyển sang schema nội bộ. Adapter kiểm
tra support, confusion matrix và metric từ prediction CSV; không suy ra thêm
thời gian, checkpoint hoặc số liệu chưa có trong artifact.

## Chạy

```bash
python -m scripts.evaluate_models \
  --cnn-predictions results/cnn/cnn_test_predictions.csv \
  --resnet18-predictions results/resnet18/resnet18_test_predictions.csv \
  --output-dir results
```

Với CNN đã bàn giao trên nhánh NV5, có thể tái tạo bảng từ metrics handoff:

```bash
python -m scripts.evaluate_models \
  --cnn-metrics results/cnn_metrics_nv2_handoff.json \
  --output-dir results
```

Khi đồng thời có prediction CSV và JSON, script tính lại metric từ CSV và từ
chối kết quả nếu Accuracy, macro metrics hoặc F1 từng lớp khác quá `1e-6`.

Với gói NV3 đã tải và kiểm SHA-256 checkpoint, dùng prediction CSV trong ZIP
và metadata handoff được version hóa trong branch NV5:

```bash
python -m scripts.evaluate_models \
  --cnn-predictions <thu_muc_nv2>/results/cnn/cnn_test_predictions.csv \
  --cnn-metrics results/cnn_metrics_nv2_handoff.json \
  --resnet18-predictions <thu_muc_nv3>/results/resnet18_test_predictions.csv \
  --resnet18-metrics results/resnet18_metrics_nv3_handoff.json \
  --output-dir results
```

Khi cột định danh là `path`, script chuẩn hóa dấu phân cách và tiền tố trước
`raw/color/`. Điều này chỉ loại khác biệt vị trí file giữa Colab và Windows;
script vẫn từ chối nếu suffix path hoặc ground truth khác nhau, hoặc có sample
ID trùng.

Đầu ra gồm:

- `results/comparison.csv`: Accuracy, macro Precision/Recall/F1, F1 từng lớp,
  số mẫu và các metadata đã bàn giao.
- `results/analysis.md`: nguồn file, kết quả, cặp lớp nhầm nhiều và giới hạn
  diễn giải.
- `results/cnn_confusion_matrix.png` và
  `results/resnet18_confusion_matrix.png` khi có matplotlib; môi trường tối
  giản dùng SVG cùng tên thay đuôi `.svg`.

Đây là bước đánh giá sau khi checkpoint đã được chọn bằng Validation. Không
dùng `comparison.csv` để chọn lại mô hình trên Test.

## Xử lý lỗi và quy tắc chống leakage

- Lỗi `không dùng cùng Test set` hoặc `Ground truth khác nhau`: dừng đánh giá,
  kiểm tra manifest/split tại NV1 và xuất lại prediction từ cùng Test DataLoader;
  không lấy giao hai CSV để tạo bảng.
- Lỗi schema, split, nhãn ngoài mapping, sample ID trùng hoặc metrics không
  khớp: sửa artifact của chủ mô hình rồi chạy lại; không sửa tay CSV/bảng kết quả.
- Thiếu checkpoint/metadata/ResNet18: chỉ để model đó `pending`. Không thay bằng
  log train, số liệu mô hình khác hoặc số minh họa.
- Dùng Validation để chọn kiến trúc, augmentation, learning rate, epoch và
  checkpoint. Chạy Test sau khi chốt lựa chọn, rồi chỉ dùng Test để báo cáo cuối.

## Checklist bàn giao cho NV5

- Gửi commit/nhánh, lệnh chạy, phiên bản Python/PyTorch/torchvision và thiết bị.
- Gửi checkpoint đã chọn, metadata checkpoint, prediction CSV và metrics JSON.
- Xác nhận manifest Test, class mapping, seed và số mẫu.
- Ghi rõ số liệu chạy lại độc lập hay lấy từ log của tác giả.
- Gửi ví dụ ảnh đúng/sai bằng `path`; không chép ảnh hoặc số liệu không có nguồn.
