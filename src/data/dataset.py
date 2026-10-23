"""Dataset đọc ảnh theo split manifest dùng chung.

Phụ trách: NV1 - Lê Xuân Bảo.

Việc cần làm (mục II.A, II.B, III tài liệu bàn giao):
- Đọc data/splits/{train,val,test}.csv (path + label + group nếu có).
- Gán nhãn theo configs/class_mapping.json; không dựa vào thứ tự alphabet
  mặc định của ImageFolder.
- Đọc ảnh và convert RGB; áp dụng transform tương ứng với từng tập.
- Gate 1: NV2 và NV3 phải load được một batch độc lập với nhãn đúng.
"""
