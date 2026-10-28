# Kết quả NV5

Thư mục này chỉ nhận các file sinh từ prediction CSV và metadata thật của NV2,
NV3. Chưa có checkpoint, manifest Test hoặc metrics trong bản repository hiện
tại nên không tạo `comparison.csv` giả.

Sau khi nhận đủ hai file prediction, chạy:

```bash
python -m scripts.evaluate_models \
  --cnn-predictions results/cnn/cnn_test_predictions.csv \
  --resnet18-predictions results/resnet18/resnet18_test_predictions.csv \
  --output-dir results
```

Các đầu ra do lệnh sinh ra gồm `comparison.csv`, `analysis.md` và hai ma trận
nhầm lẫn. Giữ nguyên prediction CSV, metrics JSON và checkpoint path để mọi số
liệu trong báo cáo/slide truy ngược được.
