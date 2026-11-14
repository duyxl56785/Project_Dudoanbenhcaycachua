# Vấn đáp ngắn phần NV4

**Vai trò:** Nguyễn Minh Duy làm giao diện [`demo/app.py`](../demo/app.py), quản lý lịch sử phiên/CSV/Dashboard trong [`demo/session.py`](../demo/session.py), API tích hợp CNN và ResNet18 tại [`src/inference.py`](../src/inference.py), kiểm tra nhất quán tại [`demo/check_consistency.py`](../demo/check_consistency.py), và HOG + Linear SVM V1/V2 tại [`src/hog_svm.py`](../src/hog_svm.py), [`results/hog_svm/`](../results/hog_svm/), [`results/hog_svm_v2/`](../results/hog_svm_v2/). HOG + HSV + RBF-SVM thuộc NV5, không phải NV4.

**Luồng demo:** Upload ảnh → kiểm tra JPG/PNG và chuyển RGB → transform Val/Test của NV1 → dựng đúng kiến trúc `build_model(4)` → nạp `state_dict` thật với `strict=True` → `eval()` và CPU inference → logits bốn lớp → softmax → nhãn, bốn xác suất, lịch sử, CSV và Dashboard. CNN resize 224×224; ResNet18 resize cạnh ngắn 256, center crop 224; cùng mean/std ImageNet. Mapping theo thứ tự `Healthy, Early Blight, Late Blight, Bacterial Spot`. Checkpoint là trọng số đã học, không phải file mã nguồn. [Hướng dẫn và hash](DEMO_README.md).

**HOG + Linear SVM:** HOG đo hướng cạnh và cấu trúc ảnh thành vector; V1 dùng ảnh xám 64×64, 1.764 đặc trưng. Scaler chỉ fit Train. Linear SVM tìm ranh giới tuyến tính giữa bốn lớp; `C` điều chỉnh mức phạt phân loại sai. Thử sáu cấu hình trên Validation và chọn `C=0.01` theo macro-F1; chỉ đánh giá Test sau khi chốt cấu hình. Train/Val/Test là 4.115/1.031/1.356 ảnh. V1 Test Accuracy 73,89%, macro-F1 69,40%; V2 dùng HOG RGB96, 900 chiều, `C=0.001`, đạt 79,13%/73,59%. CNN/ResNet18 học đặc trưng sâu nên trong thực nghiệm này cao hơn HOG tuyến tính; đây là kết quả trên bộ dữ liệu cụ thể, không phải quy luật tuyệt đối. [Bằng chứng tính lại](audits/NV4_FINAL_AUDIT_20261114.md).

**Overfitting và leakage:** Mô hình có thể học quá sát Train và kém trên ảnh mới. Nhóm tách Train/Val/Test, fit scaler trên Train, chọn `C` theo Validation, đối chiếu split không trùng. V1 đã xem Test trước khi phát triển V2, nên V2 Test là kết quả tham khảo với giới hạn độc lập; không dùng nó để chọn cấu hình. PlantVillage ảnh nền đơn giản, chưa bảo đảm độ chính xác ngoài vườn.

## 10 câu dễ nhớ

1. **Em làm gì?** Giao diện Streamlit, tích hợp suy luận hai mạng và HOG + Linear SVM.
2. **Tích hợp CNN/ResNet18 bằng gì?** PyTorch; `src/inference.py` dựng kiến trúc, strict-load checkpoint, chạy ảnh qua transform đúng mô hình.
3. **Checkpoint là gì, ở đâu?** File trọng số đã huấn luyện. CNN ở `checkpoints/cnn_best.pt`; ResNet18 lấy từ gói Drive NV3 và đặt ở `checkpoints/resnet18_best.pt`.
4. **Vì sao dùng Streamlit?** Tạo nhanh demo web Python có upload, dự đoán và biểu đồ, dễ chạy trên máy nghiệm thu.
5. **Ảnh đi qua những bước nào?** Kiểm tra ảnh → RGB → resize/crop/normalize → mô hình → softmax bốn lớp → hiển thị.
6. **Vì sao hai model có xác suất khác nhau?** Kiến trúc và trọng số học khác nhau; mỗi model tạo logits riêng, dù có thể cùng nhãn top-1.
7. **HOG là gì?** Vector mô tả hướng cạnh và hình dạng lá; V1 có 1.764 chiều.
8. **Vì sao chọn `C=0.01`?** Nó cho macro-F1 Validation tốt nhất trong sáu cấu hình V1 đã thử; không chọn bằng Test.
9. **Kết quả SVM V1?** 1.002/1.356 ảnh Test đúng, Accuracy 73,89%, macro-F1 69,40%; log fit 18,12 giây.
10. **Chứng minh đã test thế nào?** Mở PR #1/#16/#19, CSV dự đoán và log; chạy `python -m pytest -q`, `python -m demo.check_consistency <ảnh> --model both`, rồi demo ảnh thật trên Streamlit. Báo cáo mới nhất: 122 pass, 6 skip, 28 subtests từ `main`.

**Hạn chế và bước tiếp:** cần checkpoint ResNet18 bàn giao ngoài Git khi chạy máy mới, video demo cuối chưa được quay/xem lại, và cần thử thêm ảnh thực tế ngoài PlantVillage. [Báo cáo kiểm toán](audits/NV4_FINAL_AUDIT_20261114.md) ghi đầy đủ hash, confusion matrix và trạng thái GitHub.
