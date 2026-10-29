# Checklist nghiệm thu NV5 và đối chiếu artifact

## A. Checklist sản phẩm NV5

- [x] `scripts/evaluate_models.py` đọc mapping chung và hỗ trợ metrics JSON/
  prediction CSV.
- [x] Kiểm tra schema, split, duplicate/missing sample ID, ground truth và
  consistency giữa metrics JSON với prediction CSV.
- [x] `results/comparison.csv` có dòng CNN `computed` và ResNet18 `pending`,
  không điền số thiếu.
- [x] `results/analysis.md` có confusion matrix, lỗi theo cặp lớp, nguồn metrics
  và giới hạn PlantVillage.
- [x] Nội dung báo cáo theo cấu trúc MỞ ĐẦU, Phần I–IV, tài liệu tham khảo,
  phụ lục trong `report/bao_cao_nv5_framework.md`.
- [x] Nội dung slide 12 trang trong `report/slide_outline_nv5.md`.
- [x] Bộ kiểm thử NV5 và bộ câu hỏi bảo vệ được cập nhật.
- [ ] Có đủ artifact ResNet18 thật để hoàn tất so sánh hai mô hình.
- [ ] Có ảnh đúng/sai Test và ảnh demo đã được NV2/NV4 xác nhận.
- [ ] Nhóm trưởng xác nhận mẫu Word/PPTX, người trình bày và bản cuối.

## B. Đối chiếu một nguồn sự thật

| Hạng mục | File nguồn | Cách kiểm tra |
|---|---|---|
| Mapping | `configs/class_mapping.json` | 4 nhãn liên tục 0–3, tên khớp báo cáo |
| Split | artifact NV1/PR #8 | Test có 1.356 mẫu, không tự chia lại |
| CNN metrics | `results/cnn_metrics_nv2_handoff.json` | Chạy script; khớp prediction CSV dưới `1e-6` |
| Bảng | `results/comparison.csv` | Không có số CNN/ResNet khác nguồn |
| Phân tích | `results/analysis.md` | Matrix, tổng lỗi, cặp lớp khớp metrics |
| Báo cáo | `report/bao_cao_nv5_framework.md` | Chỉ dùng số đã có nguồn; TODO/PENDING rõ |
| Slide | `report/slide_outline_nv5.md` | Số và trạng thái khớp bảng/report |
| Demo | artifact NV4 | Kiểm tra checkpoint, mapping, lệnh chạy, ảnh minh họa |

## C. Artifact bắt buộc NV3 cần bàn giao

Gửi vào PR/issue hoặc đường dẫn artifact, không dán số rời:

- `results/resnet18/resnet18_test_predictions.csv` với sample ID/path, true,
  pred và split.
- `results/resnet18/resnet18_metrics.json` gồm Accuracy, macro và per-class
  metrics, confusion matrix, n_samples, checkpoint/hash, seed, device.
- Checkpoint fine-tuned tương ứng và sidecar metadata.
- Validation evidence dùng để chọn stage/checkpoint; xác nhận Test chỉ chạy sau
  khi chốt cấu hình.
- Số tham số, thời gian train/inference và phiên bản Python/PyTorch/torchvision.

## D. Cách cập nhật khi đủ ResNet18

1. Kiểm tra file prediction có cùng sample ID và ground truth với CNN.
2. Chạy:

   ```bash
   python -m scripts.evaluate_models \
     --cnn-predictions results/cnn/cnn_test_predictions.csv \
     --resnet18-predictions results/resnet18/resnet18_test_predictions.csv \
     --cnn-metrics results/cnn/cnn_metrics.json \
     --resnet18-metrics results/resnet18/resnet18_metrics.json \
     --output-dir results
   ```

3. Kiểm tra `comparison.csv`, hai confusion matrix và `analysis.md`.
4. Cập nhật report/slide từ đầu ra mới; không sửa tay một con số đơn lẻ.
5. Chạy test, ghi commit/nguồn vào PR #11 và yêu cầu NV3 + nhóm trưởng xác nhận.

## E. Người cần review/xác nhận

- **NV2 Nguyễn Văn Duy (`@duyxl56785`):** xác nhận source commit, prediction
  CSV CNN, metrics JSON và confusion matrix đã bàn giao.
- **NV3 Trần Văn Chính (`@tranvanchinhbg2006-alt`):** bàn giao và xác nhận toàn
  bộ artifact ResNet18 Test/Validation/metadata.
- **NV4 Nguyễn Minh Duy (`@nguyenminhduy090-alt`):** xác nhận lệnh chạy demo,
  checkpoint thật và ảnh minh họa.
- **Nhóm trưởng:** xác nhận cấu trúc mẫu CMC, người thuyết trình, số liệu cuối
  và review PR #11 trước khi merge.
