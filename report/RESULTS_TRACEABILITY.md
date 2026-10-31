# Đối chiếu nguồn kết quả NV5

Tài liệu này là bảng kiểm trước khi nhóm review. Mỗi số liệu trong Word và
slide hai mô hình phải đối chiếu với bảng này; không làm tròn ngược để sửa
`results/comparison.csv`.

| Nội dung báo cáo | Giá trị dùng trong báo cáo | Nguồn gốc | Kiểm tra NV5 |
| --- | --- | --- | --- |
| Split chung | Test, 1.356 ảnh | [PR #7](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/7), [PR #9](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/9), prediction CSV hai mô hình | Chuẩn hóa path rồi đối chiếu cùng sample ID và nhãn thật. |
| CNN Accuracy / macro-F1 | 0,969764 / 0,965034 | `results/cnn_metrics_nv2_handoff.json`, artifact NV2 PR #9 | Tính lại từ CSV trong `scripts/evaluate_models.py`; khớp JSON dưới 1e-6. |
| ResNet18 Accuracy / macro-F1 | 0,986726 / 0,983978 | `results/resnet18_metrics_nv3_handoff.json`, ZIP PR #7 | Tính lại từ CSV; adapter chuyển `classification_report`; khớp JSON dưới 1e-6. |
| Checkpoint ResNet18 | SHA-256 `61d44ba8...f5fdf9e2`, epoch 9, Validation macro-F1 0,980859 | ZIP bàn giao tại [PR #7](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/7) | NV5 tải ZIP và kiểm SHA-256 ngày 30/10/2026. |
| Confusion matrix và F1 từng lớp | Hai ma trận, bốn F1/lớp | `results/comparison.csv`, `results/cnn_confusion_matrix.svg`, `results/resnet18_confusion_matrix.svg` | Sinh từ prediction CSV cùng Test. |
| Minh chứng demo | CNN, ResNet18 và chế độ compare trên ảnh thật | [NV4 commit 2406b12](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/commit/2406b12a383eb5a388774442fe0b364b7e713831) | Ảnh được sao chép nguyên vẹn vào `report/assets/nv4_demo/`; không dùng làm accuracy toàn tập. |

## Quy tắc cập nhật

1. Khi checkpoint hoặc prediction CSV thay đổi, chạy lại evaluator với cả hai
   CSV và chỉ dùng output mới để cập nhật CSV, phân tích, Word và slide.
2. Không dùng Test để chọn checkpoint, learning rate hay augmentation.
3. Nếu cần ảnh dự đoán sai, NV2/NV4 phải cung cấp file hoặc link commit kèm
   nhãn thật. Không cắt hoặc gắn nhãn lại ảnh demo để tạo bằng chứng.
