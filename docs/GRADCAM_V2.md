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
- ResNet18: lớp tính Grad-CAM là đầu ra `model.layer4`.
- Gradient lấy từ logit top-1 bằng `torch.autograd.grad`, không tích lũy hoặc xóa
  `.grad` của weights. Tạm dùng eval rồi khôi phục mode từng module; luôn gỡ hook
  trong `finally`. Forward phân loại và Grad-CAM dùng chung khóa vì model được
  cache giữa các phiên Streamlit; các forward này chạy tuần tự.
- Ảnh đánh dấu là **khung hình đúng với tiền xử lý mô hình**: CNN resize
  224×224, ResNet18 resize cạnh ngắn 256 rồi crop giữa 224×224. Vùng nằm
  ngoài crop ResNet18 không hiện trên ảnh đánh dấu; ảnh gốc được đặt bên cạnh.
- Dự đoán Healthy không vẽ khung bệnh, nhưng Grad-CAM vẫn có thể hiển thị
  những vùng khiến mô hình chọn lớp Healthy.
- Softmax là xác suất tương đối giữa đúng bốn lớp, chưa hiệu chuẩn.
- Bệnh ngoài bốn lớp và ảnh không phải lá cà chua chưa được nhận biết chắc chắn.
- V1 không thay đổi: checkbox mặc định tắt; không sửa checkpoint hay transform.
- Khung dùng cụm liên thông lớn nhất có CAM ≥ 0,75 lần giá trị cực đại;
  bỏ cụm dưới max(16 pixel, 0,5% ảnh) hoặc khung chiếm ≥ 85% ảnh.
  Đây chỉ là quy tắc **hiển thị**, không phải ngưỡng phát hiện bệnh, độ tin cậy
  định vị hay bằng chứng vùng bệnh chính xác. Chưa có nhãn vị trí để kiểm định.
- Tên bệnh hiển thị Việt/Anh; ảnh gốc và overlay đặt cạnh nhau trên màn hình
  rộng, cột tự xếp dọc trên màn hình hẹp. PNG tải về là ảnh 224×224 đã crop/resize
  và phủ màu/khung như giao diện, không phải tọa độ trên ảnh gốc.
- Lỗi tạo Grad-CAM (kể cả import module) không hủy dự đoán hoặc lịch sử/CSV.

## Cập nhật giao diện (nhánh fix/v2-gradcam-ux, 06/11/2026)
Phát hiện khi chạy giao diện thật: lá Healthy đặt ở nửa trên ảnh dọc 9:16 bị cả
CNN (Late Blight 58,96%) và ResNet18 (Late Blight 76,51%) đoán sai và vẽ khung đỏ
lên lá khỏe, vì ResNet18 chỉ thấy ~43% giữa ảnh và CNN kéo méo ×1,78.
- `src/view_geometry.py`: ánh xạ vùng ResNet18 phân tích ngược về ảnh gốc theo
  đúng Resize/CenterCrop của torchvision (có test đối chiếu pixel với torchvision).
- Ảnh gốc hiện khung xanh + làm tối phần ResNet18 không phân tích; cảnh báo bằng
  lời khi ảnh dài/rộng (ResNet18 < 90% diện tích hoặc CNN méo > ×1,15).
- Grad-CAM của mỗi mô hình nằm trong cùng cột với kết quả của mô hình đó, nên
  khi màn hình hẹp xếp dọc vẫn đi đúng cặp.
- `.streamlit/config.toml`: giới hạn upload 10 MB khớp `MAX_IMAGE_BYTES`.
Không đổi checkpoint, transform, class mapping hay cách tính Grad-CAM.

## Chạy thử trên Windows

```powershell
cd "D:\CMC\HocKi5\HocMay\Project_Dudoanbenhcaycachua"
git status --short
# Chỉ switch khi đã xử lý an toàn thay đổi chưa commit:
git fetch origin
git switch feature/v2-gradcam-visualization
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m compileall -q src demo tests
.\.venv\Scripts\python.exe -m streamlit run demo/app.py --server.port 8503
```

Trong Streamlit: chọn **So sánh cả hai**, bật **Grad-CAM – thử nghiệm**,
tải 1–2 ảnh lá rõ bệnh, bấm **Dự đoán bệnh**. Kiểm tra ảnh gốc, Grad-CAM
CNN, Grad-CAM ResNet18, nhãn và bốn xác suất; thử ảnh Healthy và ảnh lỗi.
Trên CPU bật Grad-CAM sẽ chậm hơn dự đoán thuần vì cần backward pass.

Kiểm tra checkpoint thật và lưu JSON/overlay để đối chiếu (thư mục cha của
`--basetemp` cần tồn tại; pytest dùng thư mục này làm dữ liệu tạm):

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
.\.venv\Scripts\python.exe -m pytest tests/test_gradcam_checkpoints.py -q -s --basetemp .tmp/gradcam-check
$image = Get-ChildItem demo_test_images/Healthy_01* | Select-Object -First 1
.\.venv\Scripts\python.exe -m demo.check_consistency $image.FullName --model both
```

Test checkpoint tự SKIP nếu thiếu weights/ảnh; CUDA SKIP khi không khả dụng.
Test đối chiếu toàn bộ bốn xác suất trước/sau Grad-CAM, batch bật/tắt,
so sánh hai model, ảnh lỗi, weights/buffers/mode/hook và không vẽ khung cho
dự đoán Healthy. Nhãn tham chiếu lấy từ `data/splits/test.csv`; đây là kiểm tra
tích hợp trên mẫu nhỏ, không thay thế đánh giá độ chính xác toàn bộ Test.

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

## Evidence kiểm thử ngày 04/11/2026

Môi trường Windows, Python 3.14.2, torch 2.14.1+cpu, torchvision 0.29.1+cpu,
Streamlit 1.66.1. Không train lại checkpoint hoặc thay đổi kiến trúc/transform.

| Kiểm tra | Kết quả thực chạy |
|---|---|
| `.venv/Scripts/python.exe -m pytest tests -q` | 102 PASS, 3 SKIP, 21 subtests PASS; 16 cảnh báo Matplotlib/NumPy |
| `.venv/Scripts/python.exe -m compileall -q src demo tests` | PASS, exit 0 |
| `demo.check_consistency <Healthy_01> --model both` | CNN/ResNet18 PASS, delta = 0 |
| Checkpoint thật, 8 ảnh × 2 model, bốn lớp | PASS kiểm tra tích hợp; xác suất trước/sau và batch bật/tắt Grad-CAM bằng nhau tuyệt đối |
| So sánh hai model cùng ảnh, ảnh lỗi, Grad-CAM lỗi | PASS; giữ dự đoán/lịch sử, ảnh lỗi bị từ chối |
| Streamlit AppTest | PASS tên Việt/Anh, ảnh, PNG/CSV download, so sánh, lịch sử, bật/tắt và lỗi import Grad-CAM |
| `streamlit run demo/app.py --server.port 8503` | Lần khởi chạy mới FAIL vì cổng đã được dùng; xác minh tiến trình là chính lệnh này trong đúng repo, HTTP `/_stcore/health` trả `ok` |
| CUDA | 2 SKIP vì môi trường CPU; SKIP còn lại là test fallback NV1 vốn không áp dụng khi đã có dataset module |
| Screenshot trình duyệt / viewport điện thoại | SKIP: Computer Use bị chặn vì không xác định chắc chắn URL Chrome trên Windows; chưa xác nhận responsive trên thiết bị thật |

Lần pytest đầu trong sandbox có 28 lỗi fixture do không có quyền đọc thư mục
tạm `C:/Windows/Temp/pytest-of-Admin`; chạy lại ngoài sandbox mới có kết quả
PASS bên trên. Không thay đổi code để né fixture hoặc bỏ test đang lỗi.

| Nhãn trong Test manifest | Số ảnh | CNN đúng | ResNet18 đúng |
|---|---:|---:|---:|
| Healthy | 2 | 2 | 2 |
| Early Blight | 2 | 1 | 2 |
| Late Blight | 2 | 2 | 2 |
| Bacterial Spot | 2 | 2 | 2 |

CNN nhầm ảnh `Early_Blight_01_0012b9d2-2130-4a06-a834-b1f3af34f57e___RS_Erly.B 8389.JPG`
thành Late Blight ở cả chế độ bật/tắt. Cả hai ảnh Healthy không có khung đỏ
với cả hai model. Khung/heatmap đúng kích thước và hình học crop nhưng có vùng
chú ý trên nền, ví dụ CNN Early_Blight_01 và ResNet18 Bacterial_Spot_01.
Không có nhãn vị trí để kết luận khung đúng/sai theo tổn thương bệnh.
Các số 7/8 và 8/8 chỉ mô tả mẫu tích hợp này, không phải độ chính xác tổng quát.

- [JSON xác suất, nhãn, SHA-256 checkpoint và ảnh](../results/gradcam_v2/checkpoint_results.json).
- [Log pytest](../results/gradcam_v2/pytest.txt).
- [Ảnh minh chứng Grad-CAM bốn lớp](../results/gradcam_v2/attention_samples.png)
  (hình kiểm tra thuật toán, **không phải screenshot giao diện**).

Trước merge: NV2/NV3 review logit/target layer và giới hạn vùng chú ý; NV4 chạy
trình duyệt desktop/điện thoại, kiểm tra tải PNG/CSV thật và chụp screenshot.
Chưa kiểm thử ảnh ngoài PlantVillage. Checkpoint ResNet18 và sidecar hiện là
file local chưa được track; cần chủ checkpoint xác nhận nguồn/phương thức chia
sẻ trong PR. CI hiện chỉ chạy job cho nhánh `feature/nv4-demo`, nên không có
CI PASS cho branch V2. Không merge trước khi được review.
