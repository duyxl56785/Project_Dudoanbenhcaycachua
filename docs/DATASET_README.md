# Dataset & Preprocessing

Phụ trách: NV1 - Lê Xuân Bảo. Trạng thái: Chưa làm.

## 1. Nguồn dữ liệu

- Dataset / link: (chưa điền)
- Phiên bản: (chưa điền)
- Ngày tải: (chưa điền)
- Giấy phép sử dụng: (chưa điền)
- Tên thư mục/lớp gốc tương ứng với 4 lớp: (chưa điền)

## 2. Lọc lớp

(chưa điền) – cách lọc đúng 4 lớp Healthy, Early Blight, Late Blight, Bacterial Spot.

## 3. Thống kê dữ liệu

Số ảnh phải đếm lại từ dữ liệu nhóm thực sự dùng, không lấy từ bài viết trên mạng.

| Lớp            | Số ảnh gốc  | Tỉ lệ (%)   |
|----------------|-------------|-------------|
| Healthy        | (chưa điền) | (chưa điền) |
| Early Blight   | (chưa điền) | (chưa điền) |
| Late Blight    | (chưa điền) | (chưa điền) |
| Bacterial Spot | (chưa điền) | (chưa điền) |
| **Tổng**       | (chưa điền) | 100         |

- Tổng số ảnh dataset gốc: (chưa điền)
- Kích thước ảnh (rộng x cao): (chưa điền)
- Kênh màu, định dạng: (chưa điền)
- Ảnh đọc lỗi: (chưa điền)
- Biểu đồ: `figures/class_distribution.png` (chưa có)

## 4. Ảnh trùng, gần trùng, cùng lá

- Cách phát hiện (hash, perceptual hash, metadata, khảo sát mẫu): (chưa điền)
- Số ảnh trùng/gần trùng đã xử lý: (chưa điền)
- Giới hạn (nếu không có mã định danh lá): (chưa điền)

## 5. Mất cân bằng lớp

(chưa điền) – mức độ mất cân bằng, có/không cần xử lý và lý do.

## 6. Chia tập Train / Validation / Test

- Phương pháp (split chính thức theo lá hay stratified): (chưa điền)
- Tỉ lệ: (chưa điền)
- Seed: (chưa điền)
- Manifest: `data/splits/train.csv`, `val.csv`, `test.csv` (chưa có)

| Lớp            | Train       | Val         | Test        |
|----------------|-------------|-------------|-------------|
| Healthy        | (chưa điền) | (chưa điền) | (chưa điền) |
| Early Blight   | (chưa điền) | (chưa điền) | (chưa điền) |
| Late Blight    | (chưa điền) | (chưa điền) | (chưa điền) |
| Bacterial Spot | (chưa điền) | (chưa điền) | (chưa điền) |

- Kiểm tra không trùng file giữa các tập: (chưa điền)

## 7. Tiền xử lý

| Bước          | CNN         | ResNet18    |
|---------------|-------------|-------------|
| Resize        | (chưa điền) | (chưa điền) |
| Normalize     | (chưa điền) | (chưa điền) |
| Lý do         | (chưa điền) | (chưa điền) |

## 8. Augmentation (chỉ Train)

- Online hay offline: (chưa điền)
- Các phép biến đổi và lý do chọn: (chưa điền)
- Nếu offline: số ảnh trước/sau theo từng lớp: (chưa điền)
- Nếu online: số file ảnh gốc không đổi; ghi rõ phép biến đổi lấy ngẫu nhiên.
- Hình minh họa: `figures/augmentation_examples.png` (chưa có)

## 9. Cách chạy

(chưa điền) – lệnh chạy prepare/check/visualize.

## 10. Hạn chế

(chưa điền) – bao gồm khác biệt giữa ảnh PlantVillage và ảnh chụp ngoài ruộng.

## Điều kiện nghiệm thu (mục III.5)

- [ ] NV2/NV3 đọc được dữ liệu bằng script và nhãn đúng.
- [ ] Thống kê khớp đúng số file hiện có, có bảng chia tập rõ ràng.
- [ ] Không trùng file giữa các tập; rủi ro gần trùng/cùng lá được ghi nhận trung thực.
- [ ] Split và preprocessing tái tạo được.
- [ ] Augmentation ngẫu nhiên chỉ chạy trong Train.
- [ ] README đủ để người khác dùng mà không phải hỏi riêng.
