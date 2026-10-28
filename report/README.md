# Khung báo cáo và slide NV5

Thư mục này dành cho báo cáo Word/PDF và slide cuối kỳ sau khi NV2, NV3 và NV4
bàn giao đủ artifact. Hiện repository chưa có checkpoint, manifest Test hoặc
metrics thật, nên chưa tạo `.docx`/`.pptx` có số liệu. Việc tạo các tệp đó
trước khi có nguồn sẽ vi phạm quy tắc không dùng số liệu giả.

## Cấu trúc đề xuất

1. Mở đầu và mục tiêu bài toán.
2. Tổng quan nhận diện bệnh lá cà chua.
3. Phương pháp: dữ liệu, CNN, ResNet18/Transfer Learning và demo.
4. Thực nghiệm: split, preprocessing, cấu hình, giao thức đánh giá.
5. Kết quả: dùng `results/comparison.csv`, hai confusion matrix và ví dụ ảnh.
6. Kết luận, hạn chế, hướng phát triển và tài liệu tham khảo.
7. Phụ lục: lệnh chạy, phiên bản môi trường, câu hỏi bảo vệ.

Sau Gate 3, chạy `python -m scripts.evaluate_models ...`, đối chiếu số liệu
giữa `comparison.csv`, `analysis.md`, báo cáo và slide rồi mới xuất bản Word/PDF
hoặc PowerPoint.
