# V2 – Vùng chú ý Grad-CAM (bản thử nghiệm)

## Mục tiêu
Giữ nguyên hai mô hình phân loại CNN và ResNet18 cùng checkpoint V1. Khi bật
**Hiển thị vùng mô hình chú ý (Grad-CAM – thử nghiệm)** ở tab **Dự đoán**,
giao diện hiển thị ảnh gốc, ảnh đã phủ màu đỏ và đôi khi có khung đỏ quanh
một cụm vùng chú ý lớn nhất, rồi đến tên lớp và bốn xác suất.

**Khung đỏ KHÔNG phải bounding box bệnh đã được gán nhãn và kiểm định.**
Nó chỉ là vùng chú ý xấp xỉ của Grad-CAM. Vùng được tô có thể nằm ngoài
lá hoặc không trùng vùng bệnh thật. Khi bản đồ không đủ rõ, không vẽ khung.

## Giới hạn quan trọng
- Grad-CAM nhắm vào lớp **top-1** do cùng checkpoint dự đoán; không tạo lớp mới.
- CNN: lớp tính Grad-CAM là block cuối `model.features[-1]`.
- ResNet18: lớp tính Grad-CAM là `model.layer4[-1]`.
- Ảnh đánh dấu là **khung hình đúng với tiền xử lý mô hình**: CNN resize
  224×224, ResNet18 resize cạnh ngắn 256 rồi crop giữa 224×224. Vùng nằm
  ngoài crop ResNet18 không hiện trên ảnh đánh dấu; ảnh gốc được đặt bên cạnh.
- Dự đoán Healthy không vẽ khung bệnh, nhưng Grad-CAM vẫn có thể hiển thị
  những vùng khiến mô hình chọn lớp Healthy.
- Softmax là xác suất tương đối giữa đúng bốn lớp, chưa hiệu chuẩn.
- Bệnh ngoài bốn lớp và ảnh không phải lá cà chua chưa được nhận biết chắc chắn.
- V1 không thay đổi: checkbox mặc định tắt; không sửa checkpoint hay transform.

## Chạy thử trên Windows

```powershell
cd "D:\CMC\HocKi5\HocMay\Project_Dudoanbenhcaycachua"
git status --short
# Chỉ switch khi đã xử lý an toàn thay đổi chưa commit:
git fetch origin
git switch feature/v2-gradcam-visualization
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m streamlit run demo/app.py
```

Trong Streamlit: chọn **So sánh cả hai**, bật **Grad-CAM – thử nghiệm**,
tải 1–2 ảnh lá rõ bệnh, bấm **Dự đoán bệnh**. Kiểm tra ảnh gốc, Grad-CAM
CNN, Grad-CAM ResNet18, nhãn và bốn xác suất; thử ảnh Healthy và ảnh lỗi.
Trên CPU bật Grad-CAM sẽ chậm hơn dự đoán thuần vì cần backward pass.

## Điều kiện trước khi merge
1. Bộ pytest hiện có và tests/test_explainability.py PASS.
2. Chạy `demo.check_consistency --model both` với checkbox V1 tắt vẫn PASS.
3. Chụp kết quả thực tế ít nhất một ảnh mỗi lớp của bộ Test đã cố định,
   kiểm tra khung không bị đặt sai hệ tọa độ hoặc lệch ra khỏi ảnh xử lý.
4. Đánh giá trên ảnh ngoài PlantVillage một cách riêng biệt; không dùng
   độ đẹp của heatmap làm bằng chứng chẩn đoán đúng.
5. NV4 review UX và NV2/NV3 review target layer/gradients trước merge.

**Chú ý:** nhánh này chỉ thêm trực quan hóa. Không làm thay đổi độ chính xác
của CNN và ResNet18 trên ảnh ngoài thực địa.
