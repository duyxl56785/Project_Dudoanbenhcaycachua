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
