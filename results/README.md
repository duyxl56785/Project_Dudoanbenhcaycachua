# Kết quả NV5

Thư mục này chỉ nhận các file sinh từ prediction CSV và metadata thật của NV2,
NV3. `cnn_metrics_nv2_handoff.json` là bản sao có ghi nguồn từ artifact Test
đã bàn giao trong PR #9; ResNet18 vẫn để `pending` vì PR #7 chưa có metrics thật.
Không tự điền số liệu thay thế.

Sau khi nhận đủ hai file prediction, chạy:

```bash
python -m scripts.evaluate_models \
  --cnn-predictions results/cnn/cnn_test_predictions.csv \
  --resnet18-predictions results/resnet18/resnet18_test_predictions.csv \
  --output-dir results
```

Với artifact NV2 hiện có và ResNet18 chưa bàn giao, có thể tái tạo trạng thái
hiện tại bằng:

```bash
python -m scripts.evaluate_models \
  --cnn-metrics results/cnn_metrics_nv2_handoff.json \
  --output-dir results
```

Các đầu ra do lệnh sinh ra gồm `comparison.csv`, `analysis.md` và hai ma trận
nhầm lẫn. Giữ nguyên prediction CSV, metrics JSON và checkpoint path để mọi số
liệu trong báo cáo/slide truy ngược được.

Khi chỉ có metrics JSON của một model, lệnh vẫn sinh bảng với model còn thiếu ở
trạng thái `pending`; khi nhận prediction CSV của cả hai model, script kiểm tra
ID/ground truth của Test chung trước khi tính lại metrics.
