# Checklist nghiệm thu NV5 và đối chiếu artifact

## A. Checklist sản phẩm NV5

- [x] `scripts/evaluate_models.py` đọc mapping chung và hỗ trợ metrics JSON/
  prediction CSV.
- [x] Kiểm tra schema, split, duplicate/missing sample ID, ground truth và
  consistency giữa metrics JSON với prediction CSV.
- [x] `results/comparison.csv` có hai dòng `computed`; ResNet18 chỉ được điền
  sau khi NV5 tải ZIP NV3, kiểm hash checkpoint và đối chiếu CSV Test.
- [x] `results/analysis.md` có confusion matrix, lỗi theo cặp lớp, nguồn metrics
  và giới hạn PlantVillage.
- [x] Nội dung báo cáo theo cấu trúc MỞ ĐẦU, Phần I–IV, tài liệu tham khảo,
  phụ lục trong `report/bao_cao_nv5_framework.md`.
- [x] Nội dung slide 12 trang trong `report/slide_outline_nv5.md`.
- [x] Bộ kiểm thử NV5 và bộ câu hỏi bảo vệ được cập nhật.
- [x] Có artifact ResNet18 thật: checkpoint/hash, metadata, prediction CSV,
  metrics, confusion matrix, log và curve từ ZIP PR #7.
- [ ] Có ảnh đúng/sai Test và ảnh demo đã được NV2/NV4 xác nhận.
- [ ] Nhóm trưởng xác nhận mẫu Word/PPTX, người trình bày và bản cuối.

## B. Đối chiếu một nguồn sự thật

| Hạng mục | File nguồn | Cách kiểm tra |
|---|---|---|
| Mapping | `configs/class_mapping.json` | 4 nhãn liên tục 0–3, tên khớp báo cáo |
| Split | artifact NV1/PR #8 | Test có 1.356 mẫu, không tự chia lại |
| CNN metrics | `results/cnn_metrics_nv2_handoff.json` | Chạy script; khớp prediction CSV dưới `1e-6` |
| ResNet18 metrics | `results/resnet18_metrics_nv3_handoff.json` | Hash checkpoint khớp; adapter NV5 kiểm CSV/metrics dưới `1e-6` |
| Bảng | `results/comparison.csv` | Không có số CNN/ResNet khác nguồn |
| Phân tích | `results/analysis.md` | Matrix, tổng lỗi, cặp lớp khớp metrics |
| Báo cáo | `report/bao_cao_nv5_framework.md` | Chỉ dùng số đã có nguồn; TODO/PENDING rõ |
| Slide | `report/slide_outline_nv5.md` | Số và trạng thái khớp bảng/report |
| Demo | artifact NV4 | Kiểm tra checkpoint, mapping, lệnh chạy, ảnh minh họa |

## C. Artifact NV3 đã bàn giao

NV5 đã tải ZIP do PR #7 công bố, kiểm SHA-256 checkpoint
`61d44ba8ac02ee57c44cbd3835aaac2c657c97924735fea743a60df5f5fdf9e2` và nhận:

- `resnet18_test_predictions.csv`, `resnet18_metrics.json` và confusion matrix;
- checkpoint fine-tuned, sidecar metadata, config, training log, history và curve;
- Validation evidence, seed/device/version và số tham số.

Artifact không ghi thời gian train/inference, nên các trường đó vẫn để trống.

## D. Cách NV5 đã cập nhật sau bàn giao ResNet18

1. Kiểm tra SHA-256 checkpoint và chuyển schema `classification_report` của NV3
   sang schema NV5 bằng adapter, không sửa artifact gốc.
2. Chuẩn hóa tiền tố path Colab/Windows đến suffix sau `raw/color/`, kiểm tra
   cùng sample ID và ground truth với CNN.
3. Chạy:

   ```bash
   python -m scripts.evaluate_models \
     --cnn-predictions results/cnn/cnn_test_predictions.csv \
     --resnet18-predictions results/resnet18/resnet18_test_predictions.csv \
     --cnn-metrics results/cnn/cnn_metrics.json \
     --resnet18-metrics results/resnet18/resnet18_metrics.json \
     --output-dir results
   ```

4. Kiểm tra `comparison.csv`, hai confusion matrix và `analysis.md`.
5. Cập nhật report/slide từ đầu ra mới; không sửa tay một con số đơn lẻ.
6. Chạy test, ghi commit/nguồn vào PR #11 và yêu cầu nhóm trưởng xác nhận.

## E. Người cần review/xác nhận

- **NV2 Nguyễn Văn Duy (`@duyxl56785`):** xác nhận source commit, prediction
  CSV CNN, metrics JSON và confusion matrix đã bàn giao.
- **NV3 Trần Văn Chính (`@tranvanchinhbg2006-alt`):** xác nhận duy trì link ZIP
  và provenance artifact; nếu có, bổ sung thời gian train/inference.
- **NV4 Nguyễn Minh Duy (`@nguyenminhduy090-alt`):** xác nhận lệnh chạy demo,
  checkpoint thật và ảnh/video minh họa cuối.
- **Nhóm trưởng:** xác nhận cấu trúc mẫu CMC, người thuyết trình, số liệu cuối
  và review PR #11 trước khi merge.
