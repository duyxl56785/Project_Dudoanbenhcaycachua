# Kiểm toán NV4 ngày 14/11/2026

## Phạm vi và trạng thái GitHub

Kiểm tra trên worktree riêng từ `main` `d69b315b677e4bbcc600f725c7e5fd9b4203eb6f`; không sửa checkout `feature/nv4-demo` đang có thay đổi cục bộ. PR #1 (Streamlit), #7 (ResNet18), #9 (CNN), #11 (đánh giá), #16 (HOG + Linear SVM V1), #17 (so sánh) và #19 (Linear SVM V2) đã merge. PR #18 của NV5 còn mở tại `3324a47b`: CI `nv4-demo` đạt, NV4 đã [review ở HEAD này](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/18#pullrequestreview-5893470732). Điều kiện còn lại là bàn giao checkpoint ngoài Git, xóa artifact lớn khỏi branch và trưởng nhóm nghiệm thu theo [nhận xét của trưởng nhóm](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/pull/18#issuecomment-6858331694). Không tự approve hoặc merge.

## Dữ liệu, kiểm thử và demo từ `main`

- `python -m scripts.check_dataset --data-root <data/raw> --verify-hashes`: 6.502 ảnh được đối chiếu SHA-256; Train 4.115, Validation 1.031, Test 1.356, không trùng path/hash/group giữa split.
- `python -m pytest -q`: **122 passed, 6 skipped, 28 subtests passed**, không có lỗi; 16 cảnh báo deprecation. Dùng môi trường `.venv` sẵn có với checkout mã sạch, chưa kiểm thử cài mới dependencies trong venv riêng.
- `python -m compileall -q src demo scripts tests` và `git diff --check`: đạt.
- CNN `checkpoints/cnn_best.pt` 1.571.893 byte, SHA-256 `d33c95bbe15e82fa83f1fd954136ebfff6e1cbba1a86fe40af473a377390de1f`, có trong Git.
- ResNet18 fine-tuned `checkpoints/resnet18_best.pt` 44.794.251 byte, SHA-256 `61d44ba8ac02ee57c44cbd3835aaac2c657c97924735fea743a60df5f5fdf9e2`, từ [gói NV3 trên Drive](https://drive.google.com/file/d/153DOn8zLCiIhnydLoPNDcht3X0EFhKuR/view), không có trong Git. Metadata, class mapping `Healthy / Early Blight / Late Blight / Bacterial Spot` và YAML preprocessing khớp. Cả hai strict-load trên CPU, `eval()` thành công.
- Demo/API dùng chung `src/inference.py`; CNN resize 224×224, ResNet18 resize cạnh ngắn 256 và center crop 224, cả hai dùng ImageNet mean/std. Mỗi ảnh qua softmax bốn lớp. Bốn ảnh Test thật, một ảnh mỗi lớp, chạy được trên CPU. CNN dự đoán sai ví dụ Early Blight thành Late Blight; các kết quả khác trong mẫu bốn ảnh đúng với cả hai mô hình. Không dùng bốn ví dụ này làm ước lượng Accuracy.
- `python -m demo.check_consistency <ảnh Bacterial Spot Test> --model both`: CNN PASS, ResNet18 PASS, cùng nhãn Bacterial Spot, sai khác xác suất tối đa `0` giữa script và API.
- `python -m streamlit run demo/app.py --server.headless true --server.port 8514`: trình duyệt thật tải hai ảnh Test (Healthy, Bacterial Spot), preview, chọn **So sánh cả hai**, chạy dự đoán. Hai mô hình đều hiển thị nhãn đúng và đủ bốn xác suất/biểu đồ. Tab lịch sử ghi bốn lượt, Dashboard đếm hai ảnh/bốn lượt. Nút tải CSV hiện diện nhưng lượt tải trong trình duyệt điều khiển bị hủy, nên chưa xác nhận tệp CSV được tải qua trình duyệt trong lần này; các test CSV đã đạt. Các test tự động bao phủ ảnh hỏng, giới hạn batch, lịch sử và checkpoint thiếu. Ảnh trình duyệt thật đã lưu từ các phiên trước tại [`demo/screenshots/`](../../demo/screenshots/); phiên này không thêm ảnh mới. Chưa có video cuối được ghi/xem lại.

## HOG + Linear SVM V1, PR #16

Checkpoint `results/hog_svm/hog_linear_svm.joblib` SHA-256 `ec61605eb4bfd897b6ee7756e54c665dbdad6314a78ceb300d14b0e974ef974b` khớp metadata. HOG ảnh xám 64×64, vector 1.764 chiều; `StandardScaler` được fit với 4.115 ảnh Train, không fit trên Validation/Test. Sáu thử nghiệm chọn bằng Validation; cấu hình được chọn `C=0.01`, Val Accuracy `0.7449078565`, Val macro-F1 `0.6953791314`. Thời gian fit mô hình trong log `18.1233 s` (không bao gồm toàn bộ pipeline).

Tính lại từ `results/hog_svm/test_predictions.csv` đủ 1.356 dòng: **1.002 đúng / 354 sai**, Accuracy `0.7389380531` (73,89%), macro-Precision `0.6955622691`, macro-Recall `0.6972206361`, macro-F1 `0.6939609252`. Confusion matrix theo thứ tự mapping chung: `[[291,16,7,16],[32,71,68,44],[9,39,292,35],[22,35,31,348]]`. Checkpoint nạp lại và smoke inference CPU đã có tại `results/hog_svm/inference_smoke.json`; log, CSV và hướng dẫn nằm trong [`results/hog_svm/README.md`](../../results/hog_svm/README.md). PR #16 đã được reviewer `duyxl56785` approve và merge.

## Linear SVM V2, PR #19

V2 là cải tiến của NV4 trên Validation, không ghi đè V1. Checkpoint `results/hog_svm_v2/hog_linear_svm_v2.joblib` SHA-256 `756bee68d15699323c14fcbd7dc5a535c316d27c2c3d948edf0989d3d8ebc23c`; scaler fit 4.115 Train. Có 40 thử nghiệm chính và 5 thử nghiệm sơ bộ; cấu hình chọn số 22: ảnh RGB 96×96, HOG 900 chiều, `C=0.001`, `class_weight=balanced`. Val Accuracy `0.7895247333`, macro-F1 `0.7431649528`. Log ghi fit mô hình `0.964 s` và toàn bộ lưới 40 cấu hình `221.28 s`; không đánh đồng hai con số.

Tính lại Test CSV 1.356 ảnh: **1.073 đúng / 283 sai**, Accuracy `0.7912979351`, macro-Precision `0.7489017157`, macro-Recall `0.7422376690`, macro-F1 `0.7358757841`; confusion matrix `[[313,4,9,4],[40,67,69,39],[4,26,320,25],[17,25,21,373]]`. V1 đã xem Test trước khi phát triển V2; V2 chọn tham số bằng Validation nhưng Test không còn là holdout hoàn toàn độc lập cho quá trình phát triển chung. Chi tiết trong [`results/hog_svm_v2/README.md`](../../results/hog_svm_v2/README.md).

## Việc còn lại

1. Quay video demo liên tục từ `main`, mở xem lại tệp thật rồi bàn giao cho NV5; dùng [kịch bản](../DEMO_SCRIPT_NV4.md). Issue #5 tiếp tục mở cho mục này.
2. Khi triển khai trên máy sạch, tải checkpoint ResNet18 và metadata NV3, xác minh SHA-256, cài dependencies rồi chạy lại test/demo. Lần này đã dùng checkout mã sạch nhưng môi trường Python sẵn có.
3. NV5/Cường xử lý artifact PR #18 theo yêu cầu trưởng nhóm; trưởng nhóm review và quyết định merge. Không gán kết quả RBF-SVM cho mô hình Linear SVM của NV4.
