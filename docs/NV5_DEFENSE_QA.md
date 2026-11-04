# Bộ câu hỏi phản biện và đáp án chuẩn bị

Tài liệu này là khung để cả nhóm điền bằng chứng cụ thể sau Gate 3. Không
điền con số khi chưa có log hoặc file kết quả truy vết được.

1. **Vì sao dùng cùng Test split cho hai mô hình?**  Để so sánh công bằng;
   Test chỉ được dùng một lần sau khi chọn checkpoint bằng Validation.
2. **Có nguy cơ leakage không?**  Đối chiếu manifest, hash/nhóm lá nếu có,
   và báo cáo việc loại mẫu trùng giữa các split.
3. **Vì sao báo cáo Macro F1?**  Macro F1 cho trọng số ngang nhau giữa các
   lớp, nên vẫn phản ánh lớp ít mẫu khi dữ liệu mất cân bằng.
4. **CNN và ResNet18 khác nhau ở đâu?**  CNN học từ đầu; ResNet18 dùng
   backbone ImageNet rồi feature extraction và fine-tuning theo Validation.
5. **Checkpoint được chọn như thế nào?**  Theo `val_macro_f1`, tie-break bằng
   `val_loss`; không dùng Test để chọn.
6. **Cặp lớp nào dễ nhầm?**  Trả lời từ hai ma trận nhầm lẫn được xuất bởi
   `evaluate_models.py`, kèm đường dẫn ảnh ví dụ.
7. **Kết quả có áp dụng ngoài thực địa không?**  PlantVillage có điều kiện
   kiểm soát; cần dữ liệu thực địa trước khi khẳng định khả năng triển khai.
8. **Có phát hiện bệnh mới chớm không?**  Không kết luận nếu chưa có tập đánh
   giá riêng cho triệu chứng mới chớm.
9. **Ảnh không thuộc bốn lớp xử lý ra sao?**  Mô hình hiện vẫn trả một trong
   bốn lớp; đây là giới hạn cần nêu trong demo và báo cáo.
10. **Nếu số liệu trong slide khác code?**  Dừng đóng gói, truy ngược về
    prediction CSV/metrics JSON và cập nhật mọi artifact từ cùng một nguồn.

11. **Vì sao không dùng Test để chọn hyperparameter?**  Vì như vậy làm Test
    thành một phần của quá trình tối ưu, gây ước lượng lạc quan. Checkpoint và
    cấu hình phải được chốt bằng Validation trước.
12. **Overfitting nhận biết và xử lý thế nào?**  So sánh loss/metric Train với
    Validation theo epoch, dùng augmentation/regularization/early stopping nếu
    đã được chủ mô hình ghi nhận. Không chọn theo Test.
13. **Transfer Learning có ý nghĩa gì?**  Dùng đặc trưng đã học từ dữ liệu lớn
    làm khởi tạo, đóng băng backbone ở feature extraction rồi fine-tune có kiểm
    soát. Hiệu quả thực tế của ResNet18 chỉ kết luận sau khi có Test artifact.
14. **Vì sao Accuracy không đủ?**  Accuracy có thể che khuất lớp ít mẫu; macro
    Precision/Recall/F1 cho trọng số ngang nhau giữa bốn lớp. Cần đọc cùng
    confusion matrix và support từng lớp.
15. **Đọc confusion matrix ra sao?**  Hàng là nhãn thật, cột là nhãn dự đoán;
    đường chéo là dự đoán đúng, ô ngoài đường chéo là hướng nhầm. Với CNN hiện
    có, Early Blight/Late Blight là cặp nhầm nhiều nhất.
16. **PlantVillage có đại diện ảnh ngoài thực địa không?**  Không đủ cơ sở để
    khẳng định. Ảnh PlantVillage tương đối kiểm soát; cần kiểm định ngoài miền
    trước khi triển khai.
17. **Demo có phải bằng chứng mô hình tốt không?**  Không. Demo kiểm tra luồng
    upload/inference/UI; hiệu năng phải lấy từ Test chung và artifact đã truy vết.
18. **Nếu ResNet18 chưa có metrics thì trình bày thế nào?**  Ghi `PENDING`, nêu
    dependency cụ thể cần NV3 bàn giao, không thay bằng số random hoặc số từ
    mô hình khác.
19. **Nếu hai prediction CSV khác sample ID thì làm gì?**  Dừng đánh giá; sửa
    nguồn bàn giao hoặc xác nhận lại manifest/split. Không cắt giao nhau để tạo
    bảng so sánh thuận tiện.
20. **Ai chịu trách nhiệm xác nhận cuối?**  NV2 xác nhận CNN, NV3 xác nhận
    ResNet18, NV4 xác nhận demo; nhóm trưởng duyệt cấu trúc và PR trước merge.
