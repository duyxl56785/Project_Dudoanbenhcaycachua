"""Chuẩn bị dữ liệu và tạo split Train/Validation/Test.

Phụ trách: NV1 - Lê Xuân Bảo.

Việc cần làm (mục II.A, II.B, III tài liệu bàn giao):
- Lọc đúng 4 lớp từ PlantVillage; ghi link, phiên bản, ngày tải, giấy phép.
- Kiểm tra ảnh trùng/gần trùng (hash, perceptual hash, metadata) trước khi
  chia; ảnh cùng lá (nếu xác định được) không nằm ở hai tập khác nhau.
- Chia từ ảnh gốc TRƯỚC augmentation; ưu tiên split chính thức theo lá nếu
  có, nếu không cân nhắc 70/15/15 stratified theo lớp (chưa chốt).
- Lưu seed và manifest data/splits/{train,val,test}.csv.
- Xuất configs/class_mapping.json và data/dataset_summary.csv.
"""
