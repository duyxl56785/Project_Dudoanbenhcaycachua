# Năm câu hỏi bảo vệ TV5

Tài liệu này tóm tắt năm câu hỏi về phần đánh giá và báo cáo. Số liệu dùng
trong câu trả lời lấy từ `results/comparison.csv` và artifact đã được NV5
đối chiếu.

## 1. Tại sao cần Accuracy, Precision, Recall và F1-score?

Accuracy cho biết tỷ lệ dự đoán đúng trên toàn bộ Test. Precision cho biết
trong các ảnh mô hình dự đoán là một lớp, có bao nhiêu ảnh đúng lớp đó. Recall
cho biết mô hình tìm được bao nhiêu ảnh thực sự thuộc lớp đó. F1-score cân bằng
Precision và Recall. Vì bốn lớp Test có số ảnh khác nhau, báo cáo dùng
macro-F1 để mỗi lớp có trọng số như nhau, thay vì chỉ nhìn Accuracy.

## 2. Confusion Matrix giúp phân tích lỗi như thế nào?

Confusion Matrix ghi số ảnh theo cặp nhãn thật và nhãn dự đoán, nên chỉ ra mô
hình nhầm lớp nào với lớp nào. Với CNN, lỗi Early Blight thành Late Blight có
14 ảnh. Với ResNet18, cặp này còn 6 ảnh. Vì vậy ma trận nhầm lẫn giúp tìm lỗi
cụ thể mà một chỉ số tổng quát như Accuracy không thể hiện.

## 3. Tại sao hai mô hình phải dùng cùng tập Test?

So sánh chỉ công bằng khi CNN và ResNet18 dự đoán trên cùng ảnh và cùng nhãn
thật. Nếu dùng hai Test set khác nhau, chênh lệch có thể do độ khó dữ liệu thay
vì do mô hình. `scripts/evaluate_models.py` kiểm tra sample ID và ground truth
trước khi so sánh. Cả hai artifact trong dự án có 1.356 ảnh Test.

## 4. Vì sao ResNet18 đạt kết quả cao hơn CNN trong thử nghiệm này?

Trên cùng Test, ResNet18 đạt Accuracy 98,67% và macro-F1 98,40%, trong khi CNN
đạt 96,98% và 96,50%. ResNet18 dùng trọng số ImageNet pretrained và fine-tuning,
nên có thể học đặc trưng ảnh tốt hơn CNN học từ đầu. Đây là giải thích kỹ thuật
hợp lý, nhưng không khẳng định nguyên nhân tuyệt đối vì dự án chưa lặp lại nhiều
seed hoặc đánh giá ngoài PlantVillage.

## 5. Làm thế nào để kiểm chứng kết quả đánh giá là chính xác?

NV5 kiểm schema CSV, mapping bốn lớp, split Test, ID trùng và nhãn thật giữa
hai prediction CSV. Script tính lại Accuracy, macro Precision, Recall, F1,
F1 từng lớp và Confusion Matrix, rồi đối chiếu metrics JSON với sai số tối đa
`1e-6`. Checkpoint ResNet18 cũng đã đối chiếu SHA-256. Kết quả chỉ được dùng
sau khi checkpoint đã chọn bằng Validation, không dùng Test để chỉnh mô hình.
