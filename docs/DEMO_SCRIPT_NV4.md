# Kịch bản video demo NV4 – Nguyễn Minh Duy

Thời lượng gợi ý: 4–6 phút. Chỉ quay khi đang dùng checkpoint nhóm xác nhận; không gọi ResNet18 là đã hoạt động nếu chưa có checkpoint fine-tuned.

## 1. Mở đầu

**Màn hình:** mở trang đầu Streamlit và giữ phần tiêu đề trong khung hình.

**Lời nói:** “Xin chào thầy cô và các bạn. Đây là hệ thống dự đoán bệnh lá cà chua của nhóm, phân loại ảnh vào bốn lớp Healthy, Early Blight, Late Blight và Bacterial Spot.”

## 2. Vai trò NV4

**Màn hình:** trỏ vào phần giới thiệu ứng dụng.

**Lời nói:** “Em là Nguyễn Minh Duy, phụ trách NV4: xây dựng giao diện Streamlit, tích hợp pipeline suy luận chung cho CNN và ResNet18, kiểm tra đầu vào, kiểm thử và chuẩn bị bằng chứng demo.”

## 3. Giới thiệu giao diện

**Màn hình:** lần lượt chỉ vùng chọn mô hình, thiết bị, uploader và nút **Dự đoán bệnh**.

**Lời nói:** “Người dùng có thể chọn CNN, ResNet18 hoặc so sánh cả hai; chọn thiết bị tự động, CPU hoặc CUDA; sau đó tải tối đa tám ảnh JPG hoặc PNG, mỗi ảnh tối đa 10 MiB.”

## 4. Demo tải ảnh thật

**Màn hình:** chọn CNN, tải ảnh `00416648-...3110.JPG` từ lớp Bacterial Spot và đợi preview xuất hiện.

**Lời nói:** “Ứng dụng kiểm tra nội dung ảnh thật, không chỉ dựa vào phần mở rộng. Ảnh hợp lệ được chuyển về RGB và hiển thị preview trước khi suy luận.”

## 5. Demo dự đoán

**Màn hình:** bấm **Dự đoán bệnh**, cuộn đến bảng và biểu đồ.

**Lời nói:** “CNN dự đoán ảnh này là Bacterial Spot. Bảng hiển thị đủ bốn xác suất: Bacterial Spot khoảng 96,95%, Early Blight 2,99%, Late Blight 0,06% và Healthy gần 0%. Biểu đồ giúp so sánh trực quan. Đây là softmax, chưa phải xác suất chẩn đoán đã hiệu chỉnh.”

## 6. Giải thích pipeline tích hợp

**Màn hình:** có thể mở sơ đồ luồng trong `docs/DEMO_README.md`.

**Lời nói:** “Giao diện và script dùng chung `src/inference.py`. Ảnh đi qua transform Val/Test của NV1, mô hình được dựng bằng `build_model(num_classes=4)`, checkpoint plain state_dict được strict-load, rồi logits bốn lớp được chuyển thành softmax. Metadata bắt buộc phải khớp nhãn và preprocessing.”

## 7. Demo ảnh không hợp lệ

**Màn hình:** tải hai ảnh thật thuộc lớp Bacterial Spot và Healthy cùng file TXT đổi tên thành `corrupt.jpg`; bấm **Dự đoán bệnh**.

**Lời nói:** “Hai ảnh thật được xử lý độc lập; một ảnh trả về Bacterial Spot, ảnh còn lại Healthy. File hỏng được báo lỗi riêng nên không làm dừng hai kết quả thật. File rỗng, GIF và file quá 10 MiB cũng bị chặn.”

## 8. Lịch sử, CSV và dashboard

**Màn hình:** mở tab **Lịch sử phiên**, chọn lần lượt một kết quả thành công và lượt lỗi, bấm **Tải kết quả CSV**; mở file trong Excel nếu có. Sang **Thống kê phiên**, quan sát hai ảnh thành công và một lượt lỗi; thử bộ lọc mô hình, quay lại lịch sử và bấm **Xóa lịch sử**.

**Lời nói:** “Lịch sử chỉ giữ tên ảnh, thời điểm và kết quả trong phiên, tối đa 200 bản ghi; không lưu nội dung ảnh. CSV UTF-8 BOM chứa trạng thái và xác suất của bốn lớp, còn lượt lỗi để trống xác suất. Dashboard đếm lượt thành công, lượt lỗi, nhãn mô hình dự đoán và thời gian forward đo thật. Đây không phải tỷ lệ bệnh thực tế ngoài vườn.”

## 9. Trạng thái ResNet18

**Màn hình:** tải lại ảnh thật, chọn ResNet18 rồi bấm dự đoán; sau đó chọn **So sánh cả hai**.

**Lời nói:** “Hiện NV3 chưa bàn giao checkpoint ResNet18 fine-tuned. Ứng dụng báo rõ thiếu checkpoint và tuyệt đối không dùng trọng số ngẫu nhiên để tạo kết quả giả. Ở chế độ so sánh, CNN vẫn chạy bình thường còn lỗi ResNet18 được hiển thị riêng.”

## 10. Kết quả kiểm thử

**Màn hình:** mở bảng kiểm thử trong `docs/DEMO_README.md`.

**Lời nói:** “Bộ test tự động hiện có 34 test và 19 subtests đều đạt. Trình duyệt đã kiểm tra đủ bốn lớp. Ba ảnh được dự đoán đúng; ảnh Early Blight đầu tiên bị nhầm thành Late Blight với 58,16% so với 41,74%. Nhóm ghi nhận nguyên kết quả này vì báo cáo Test của CNN cũng cho thấy nhầm lẫn chính giữa hai lớp bệnh này.”

## 11. Giới hạn và kết luận

**Màn hình:** quay lại phần cảnh báo đầu trang.

**Lời nói:** “Hệ thống chỉ biết bốn lớp đã học, dữ liệu PlantVillage chủ yếu ở điều kiện kiểm soát và kết quả không thay thế chuyên gia. Phần độc lập của NV4 và tích hợp CNN thật đã được kiểm chứng. Gate 4 vẫn chờ checkpoint ResNet18 fine-tuned, metadata và kết quả Validation từ NV3. Em xin cảm ơn.”

## Checklist trước khi quay

- Dùng đúng branch/commit đã review và checkpoint có hash xác nhận.
- Tắt thông báo cá nhân, không quay đường dẫn chứa thông tin nhạy cảm.
- Chuẩn bị sẵn một ảnh thật, một file hỏng và cửa sổ README.
- Quay rõ preview, tên bệnh, đủ bốn xác suất, biểu đồ và thông báo lỗi.
- Quay lịch sử có cả lượt lỗi, tải CSV, dashboard và thao tác xóa lịch sử.
- Không cắt ghép để thay đổi nhãn hoặc xác suất; nếu dự đoán sai, giữ nguyên và giải thích.
- Sau khi NV3 bàn giao checkpoint thật, quay bổ sung ResNet18 và chế độ so sánh; trước đó phải nói rõ đang thiếu checkpoint.

**Trạng thái:** chưa có video được ghi hoặc bàn giao; các ảnh chụp trình duyệt thật nằm trong `demo/screenshots/`.

## Hướng dẫn quay video CNN thật cho NV5

1. Tạo môi trường theo `docs/DEMO_README.md`; dùng đồng bộ mã, YAML, checkpoint và metadata CNN từ PR #9. Đối chiếu SHA-256 `cnn_best.pt` với `d33c95bbe15e82fa83f1fd954136ebfff6e1cbba1a86fe40af473a377390de1f` trước khi quay. Chạy `streamlit run demo/app.py` từ gốc dự án.
2. Mở demo trong một cửa sổ trình duyệt riêng. Trong phần mềm quay, chọn **Window Capture / quay cửa sổ ứng dụng** cho đúng cửa sổ này, không chọn toàn màn hình. Quay thử vài giây rồi xem lại để chắc chắn khung hình chỉ chứa demo và chữ đọc được; tắt thông báo cá nhân.
3. Dùng ảnh thật trong `data/raw/` nhưng không đưa nguyên dataset vào video. Quay liên tục thao tác upload, preview, chọn CNN, dự đoán và bảng bốn xác suất. Sau đó thêm ảnh hỏng, mở lịch sử, tải CSV, xem dashboard, rồi chọn chế độ so sánh để ghi rõ ResNet18 chưa có checkpoint. Giữ nguyên mọi kết quả thật, kể cả dự đoán sai.
4. Kiểm tra file MP4 sau khi dừng: hình và tiếng (nếu thuyết minh), tên bệnh, bốn xác suất, thông báo ảnh hỏng và trạng thái ResNet18 phải xem được. Chỉ đánh dấu video hoàn thành trong Issue khi tệp đã được mở và đối chiếu. Không tuyên bố đã chạy ResNet18 trước khi NV3 bàn giao checkpoint thật.

NV4 đã thử khả năng quay màn hình ngày 29/10/2026 nhưng chế độ ghi desktop thu cả ứng dụng ngoài dự án, nên không dùng bản quay đó làm bằng chứng. Ảnh trình duyệt thật và kịch bản hiện có đủ để NV5 dựng phần CNN; video thật vẫn chưa được bàn giao.
