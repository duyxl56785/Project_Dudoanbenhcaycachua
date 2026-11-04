# Minh chứng giao diện Streamlit từ NV4

Ba ảnh JPEG trong thư mục này được sao chép **nguyên vẹn** từ branch
`feature/nv4-demo`, commit
[`2406b12`](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/commit/2406b12a383eb5a388774442fe0b364b7e713831), để nhúng vào sản phẩm báo cáo
của NV5. Chúng không phải ảnh đánh giá toàn bộ Test set và không thay thế
prediction CSV/metrics.

| File NV5 | Nguồn NV4 | Nội dung được phép mô tả |
| --- | --- | --- |
| `cnn_prediction_real.jpg` | [`demo/screenshots/cnn_prediction_real.jpg`](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/blob/2406b12a383eb5a388774442fe0b364b7e713831/demo/screenshots/cnn_prediction_real.jpg) | Một lượt demo CNN dự đoán Bacterial Spot 96,95%. |
| `resnet18_real_20261030.jpg` | [`demo/screenshots/resnet18_real_20261030.jpg`](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/blob/2406b12a383eb5a388774442fe0b364b7e713831/demo/screenshots/resnet18_real_20261030.jpg) | Một lượt demo ResNet18 dự đoán Bacterial Spot 67,11%. |
| `cnn_resnet18_compare_20261030.jpg` | [`demo/screenshots/cnn_resnet18_compare_20261030.jpg`](https://github.com/duyxl56785/Project_Dudoanbenhcaycachua/blob/2406b12a383eb5a388774442fe0b364b7e713831/demo/screenshots/cnn_resnet18_compare_20261030.jpg) | Chế độ so sánh hai mô hình trên cùng một ảnh. |

NV4 chưa bàn giao ảnh chụp riêng cho một dự đoán **sai** có nhãn thật hiển thị.
Vì vậy NV5 chỉ dùng các ảnh đúng ở trên; số lỗi và các cặp dễ nhầm trong báo
cáo lấy từ prediction CSV đã kiểm chứng.
