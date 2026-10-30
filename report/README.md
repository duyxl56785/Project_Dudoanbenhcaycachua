# Khung báo cáo và slide NV5

Thư mục này dành cho báo cáo Word/PDF và slide cuối kỳ sau khi NV2, NV3 và NV4
bàn giao đủ artifact. Hiện repository chưa có checkpoint, manifest Test hoặc
metrics thật, nên chưa tạo `.docx`/`.pptx` có số liệu. Việc tạo các tệp đó
trước khi có nguồn sẽ vi phạm quy tắc không dùng số liệu giả.

## Thư mục

- `deliverables/`: các tệp nộp hoặc tệp trình chiếu (`.pptx`, `.docx`).
- `tools/`: script tạo các tệp trong `deliverables/`; không đặt lẫn với code mô
  hình, dữ liệu hay script đánh giá.
- Thư mục hiện tại: nội dung Markdown, checklist và tài liệu nguồn NV5.

## Cấu trúc đề xuất

1. Mở đầu và mục tiêu bài toán.
2. Tổng quan nhận diện bệnh lá cà chua.
3. Phương pháp: dữ liệu, CNN, ResNet18/Transfer Learning và demo.
4. Thực nghiệm: split, preprocessing, cấu hình, giao thức đánh giá.
5. Kết quả: dùng `results/comparison.csv`, hai confusion matrix và ví dụ ảnh.
6. Kết luận, hạn chế, hướng phát triển và tài liệu tham khảo.
7. Phụ lục: lệnh chạy, phiên bản môi trường, câu hỏi bảo vệ.

Hiện đã có khung slide NV5 và bảng CNN từ artifact bàn giao của NV2; ResNet18
được ghi `pending` cho tới khi NV3 bàn giao metrics thật. Sau Gate 3, chạy
`python -m scripts.evaluate_models ...`, đối chiếu số liệu giữa
`comparison.csv`, `analysis.md`, báo cáo và slide rồi mới xuất bản Word/PDF
hoặc PowerPoint.

Khung nội dung có thể điền vào báo cáo cuối kỳ nằm ở
[`bao_cao_nv5_framework.md`](bao_cao_nv5_framework.md), gồm giới thiệu, dataset,
phương pháp, quy trình thí nghiệm, CNN, đánh giá, hạn chế và nguồn truy vết.

Repository chỉ có hướng dẫn cấu trúc trong tài liệu bàn giao, chưa có file mẫu
Word/PPTX chính thức do trường xác nhận. Vì vậy bản này cung cấp nội dung nguồn,
không tự nhận là đúng quy cách nộp cuối kỳ. Outline slide 12 trang nằm ở
[`slide_outline_nv5.md`](slide_outline_nv5.md); checklist nghiệm thu nằm ở
[`NV5_ACCEPTANCE_CHECKLIST.md`](NV5_ACCEPTANCE_CHECKLIST.md).
