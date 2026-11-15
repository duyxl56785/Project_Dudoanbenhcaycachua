# Tệp nộp và trình chiếu

Thư mục này chỉ chứa tệp Word và PowerPoint phục vụ báo cáo hoặc trình bày.
Các script sinh tệp được đặt trong `../tools/`; nội dung và checklist NV5 ở
thư mục `report/` cha. Không đặt mã nguồn mô hình, dữ liệu hoặc kết quả đánh
giá vào đây.

- `bao_cao_nhom_tomato_template.docx` là khung cũ chưa có kết quả.
- `bao_cao_nhom_tomato_draft_nv5.docx` là bản Word chỉnh sửa được có kết quả
  CNN đã kiểm chứng và ghi rõ ResNet18 `PENDING`; chưa phải bản nộp cuối.
- `slides_nhom_tomato_draft_nv5.pptx` là slide chỉnh sửa được có số liệu CNN
  đã kiểm chứng; các ô ResNet18 giữ trạng thái `PENDING`.
- `bao_cao_nhom_tomato_two_models_nv5.docx` và
  `slides_nhom_tomato_two_models_nv5.pptx` có kết quả CNN/ResNet18 đã kiểm chứng
  trên cùng Test. Bản cập nhật có bảng so sánh, biểu đồ Macro-F1, confusion
  matrix và ảnh demo Streamlit có nguồn NV4; chưa thay thế mẫu nộp CMC chính
  thức. Đối chiếu nguồn tại `../RESULTS_TRACEABILITY.md`.

## Bản nộp cuối (15/11/2026)

- `bao_cao_nhom_tomato_final.docx`: báo cáo nộp, gồm CNN, ResNet18 và hai
  baseline HOG + SVM (NV4, NV5); số liệu khớp `results/comparison.csv`.
- `slides_bao_cao_cong_viec_final.pptx`: slide báo cáo công việc theo mẫu
  CMC (1 trang giới thiệu, 5 trang từng thành viên, 1 trang tổng kết). Cột
  "Tự đánh giá" và "Trưởng nhóm đánh giá" do nhóm điền khi trình bày.
- Các tệp `*_nv5`, `*_draft_*`, `*_template*` ở trên là bản trung gian, giữ lại
  để truy vết.
