# Kịch bản demo trực tiếp NV4 – Nguyễn Minh Duy

Thời lượng gợi ý: 4–6 phút. Dùng HEAD `main` mới nhất, lấy checkpoint ResNet18 thật của NV3 và xác minh SHA-256 của cả hai mô hình trước buổi bảo vệ. PR #1/#7/#9 đã merge. Nhóm thống nhất trình bày demo trực tiếp, không yêu cầu quay video. Không trình bày ảnh chụp cũ có thông báo thiếu checkpoint như trạng thái hiện tại.

## 1. Mở đầu

**Thao tác:** mở trang đầu Streamlit và giới thiệu tiêu đề ứng dụng.

**Lời nói:** “Xin chào thầy cô và các bạn. Đây là hệ thống dự đoán bệnh lá cà chua của nhóm, phân loại ảnh vào bốn lớp Healthy, Early Blight, Late Blight và Bacterial Spot.”

## 2. Vai trò NV4

**Thao tác:** giới thiệu phần mô tả ứng dụng.

**Lời nói:** “Em là Nguyễn Minh Duy, phụ trách NV4: xây dựng giao diện Streamlit, tích hợp pipeline suy luận chung cho CNN và ResNet18, kiểm tra đầu vào, kiểm thử và chuẩn bị bằng chứng demo.”

## 3. Giới thiệu giao diện

**Thao tác:** lần lượt chỉ vùng chọn mô hình, thiết bị, uploader và nút **Dự đoán bệnh**.

**Lời nói:** “Người dùng có thể chọn CNN, ResNet18 hoặc so sánh cả hai; chọn thiết bị tự động, CPU hoặc CUDA; sau đó tải tối đa tám ảnh JPG hoặc PNG, mỗi ảnh tối đa 10 MiB.”

## 4. Demo tải ảnh thật

**Thao tác:** chọn CNN, tải ảnh `00416648-...3110.JPG` từ lớp Bacterial Spot và đợi preview xuất hiện.

**Lời nói:** “Ứng dụng kiểm tra nội dung ảnh thật, không chỉ dựa vào phần mở rộng. Ảnh hợp lệ được chuyển về RGB và hiển thị preview trước khi suy luận.”

## 5. Demo dự đoán

**Thao tác:** bấm **Dự đoán bệnh**, cuộn đến bảng và biểu đồ.

**Lời nói:** “CNN dự đoán ảnh này là Bacterial Spot. Bảng hiển thị đủ bốn xác suất: Bacterial Spot khoảng 96,95%, Early Blight 2,99%, Late Blight 0,06% và Healthy gần 0%. Biểu đồ giúp so sánh trực quan. Đây là softmax, chưa phải xác suất chẩn đoán đã hiệu chỉnh.”

## 6. Giải thích pipeline tích hợp

**Thao tác:** mở sơ đồ luồng trong `docs/DEMO_README.md` nếu cần giải thích.

**Lời nói:** “Giao diện và script dùng chung `src/inference.py`. Ảnh đi qua transform Val/Test của NV1, mô hình được dựng bằng `build_model(num_classes=4)`, checkpoint plain state_dict được strict-load, rồi logits bốn lớp được chuyển thành softmax. Metadata bắt buộc phải khớp nhãn và preprocessing.”

## 7. Demo ResNet18 và so sánh hai mô hình

**Thao tác:** giữ ảnh Bacterial Spot, chọn ResNet18 và bấm **Dự đoán bệnh**; sau đó chọn **So sánh cả hai** và chạy lại cùng ảnh. Chỉ rõ bảng và biểu đồ bốn xác suất của mỗi mô hình.

**Lời nói:** “Checkpoint ResNet18 fine-tuned của NV3 đã được xác minh và nạp nghiêm ngặt trên CPU. Trên ảnh này, ResNet18 dự đoán Bacterial Spot khoảng 67,11%; CNN dự đoán cùng nhãn khoảng 96,95%. Hai bảng xác suất thể hiện đầu ra riêng của từng mô hình. Đây là ví dụ minh họa, không phải phép đánh giá độ chính xác toàn tập.”

## 8. Demo nhiều ảnh và ảnh không hợp lệ

**Thao tác:** trong chế độ so sánh, tải hai ảnh thật thuộc lớp Bacterial Spot và Healthy cùng file TXT đổi tên thành `corrupt.jpg`; bấm **Dự đoán bệnh**. Chỉ rõ hai ảnh hợp lệ có kết quả CNN/ResNet18 riêng và ảnh hỏng có lỗi riêng.

**Lời nói:** “Hai ảnh thật được xử lý độc lập; một ảnh trả về Bacterial Spot, ảnh còn lại Healthy ở cả hai mô hình. File hỏng được báo lỗi riêng nên không làm dừng các kết quả thật. File rỗng, GIF và file quá 10 MiB cũng bị chặn.”

## 9. Lịch sử, CSV và dashboard

**Thao tác:** mở tab **Lịch sử phiên**, chọn lần lượt một kết quả CNN, một kết quả ResNet18 và lượt lỗi; bấm **Tải kết quả CSV** và mở file nếu có. Sang **Thống kê phiên**, chỉ rõ tổng ảnh, lượt theo mô hình và lượt lỗi; thử bộ lọc mô hình. Xóa lịch sử sau khi trình bày xong.

**Lời nói:** “Lịch sử chỉ giữ tên ảnh, thời điểm và kết quả trong phiên, tối đa 200 bản ghi; không lưu nội dung ảnh. CSV UTF-8 BOM chứa trạng thái và xác suất của bốn lớp, còn lượt lỗi để trống xác suất. Dashboard đếm lượt thành công, lượt lỗi, nhãn mô hình dự đoán và thời gian forward đo thật. Đây không phải tỷ lệ bệnh thực tế ngoài vườn.”

## 10. Kết quả kiểm thử

**Thao tác:** mở bảng kiểm thử trong `docs/DEMO_README.md`.

**Lời nói:** “Từ checkout sạch của `main` ngày 21/11, bộ kiểm thử đạt 130 test, 7 skip và 28 subtests. Riêng HOG V1/V2 có 12 test đạt. Hai checkpoint thật strict-load trên CPU. Script và API cho CNN/ResNet18 khớp xác suất trên cùng ảnh, sai khác tối đa bằng 0. Nhóm giữ nguyên kết quả thật, kể cả dự đoán sai.”

## 11. Giới hạn và kết luận

**Thao tác:** chỉ lại phần cảnh báo đầu trang.

**Lời nói:** “Hệ thống chỉ biết bốn lớp đã học, dữ liệu PlantVillage chủ yếu ở điều kiện kiểm soát và kết quả không thay thế chuyên gia. CNN và ResNet18 đã được kiểm thử từ `main` với checkpoint thật. Checkpoint ResNet18 cần tải riêng từ gói NV3 khi chạy trên máy khác. Em xin cảm ơn.”

## Chuẩn bị trước buổi bảo vệ

- Dùng HEAD `main` mới nhất; checkpoint CNN SHA-256 `d33c95bb...f473a377390de1f`, ResNet18 SHA-256 `61d44ba8...f5f5fdf9e2`.
- Đóng các cửa sổ không liên quan và chuẩn bị trình duyệt mở ứng dụng.
- Chuẩn bị sẵn một ảnh thật, một file hỏng và cửa sổ README.
- Trình bày preview, nhãn bệnh, đủ bốn xác suất và biểu đồ của CNN/ResNet18, chế độ so sánh và thông báo ảnh lỗi.
- Mở lịch sử có cả lượt lỗi, tải CSV, dashboard và thao tác xóa lịch sử.
- Giữ nguyên mọi dự đoán, kể cả dự đoán sai; giải thích giới hạn của dữ liệu PlantVillage.

**Trạng thái 21/11/2026:** nhóm thống nhất demo trực tiếp khi bảo vệ, không yêu cầu video (commit `03e69f6` trên `main`). Ảnh chụp trình duyệt thật của hai mô hình vẫn nằm trong `demo/screenshots/` để tham khảo.

## Trình tự demo trực tiếp hai mô hình

1. Tạo môi trường theo `docs/DEMO_README.md` từ `main`; đặt checkpoint/metadata ResNet18 NV3 vào `checkpoints/`. Đối chiếu SHA-256 CNN `d33c95bbe15e82fa83f1fd954136ebfff6e1cbba1a86fe40af473a377390de1f`, ResNet18 `61d44ba8ac02ee57c44cbd3835aaac2c657c97924735fea743a60df5f5fdf9e2` trước buổi bảo vệ. Chạy `python -m streamlit run demo/app.py` từ gốc dự án.
2. Mở demo trong trình duyệt và giữ cửa sổ ở trạng thái sẵn sàng. Chuẩn bị ảnh thật thuộc ít nhất hai lớp và một file ảnh lỗi.
3. Trình bày upload, preview, CNN, ResNet18, so sánh hai mô hình, bốn xác suất, ảnh lỗi, lịch sử, tải CSV và Dashboard. Giữ nguyên mọi kết quả thật, kể cả dự đoán sai.
4. Kết thúc bằng giới hạn: mô hình chỉ nhận bốn lớp đã học; dữ liệu PlantVillage chủ yếu chụp trong điều kiện kiểm soát; kết quả không thay thế chuyên gia nông nghiệp.

Không cần tạo hoặc bàn giao video. Dùng ảnh trong `demo/screenshots/` và trình bày ứng dụng trực tiếp theo trình tự trên.
