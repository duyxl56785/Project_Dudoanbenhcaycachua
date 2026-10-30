# Báo cáo bài tập lớn: Nhận diện bệnh qua ảnh lá cà chua

> Bản nội dung nguồn do NV5 tổng hợp theo cấu trúc được ghi trong
> `docs/BAN_GIAO_NHIEM_VU_NHOM_HOC_MAY_CA_CHUA_23-10-2026.txt`. Đây chưa phải
> bản Word đã xác nhận với giảng viên; nhóm cần kiểm tra mẫu bìa, font, lề,
> đánh số bảng/hình và yêu cầu PDF trước khi nộp.

## MỞ ĐẦU

### Lý do chọn đề tài

Nhận dạng bệnh trên lá cà chua từ ảnh có thể hỗ trợ sàng lọc nhanh trong quy
trình chăm sóc cây. Bài toán trong phạm vi này là phân loại ảnh vào các lớp đã
được thống nhất, không thay thế chẩn đoán chuyên gia và không khẳng định khả
năng phát hiện bệnh mới chớm.

### Mục tiêu

1. Chuẩn hóa dữ liệu, mapping nhãn và split dùng chung cho cả nhóm.
2. Xây dựng CNN học từ đầu và khảo sát hướng ResNet18 Transfer Learning.
3. Đánh giá hai mô hình trên cùng Test split sau khi checkpoint đã được chọn
   bằng Validation.
4. Tích hợp mô hình vào demo và tổng hợp kết quả có thể truy vết từ artifact.

### Phạm vi và câu hỏi nghiên cứu

Phạm vi gồm bốn lớp `Healthy`, `Early Blight`, `Late Blight`, `Bacterial Spot`.
Câu hỏi chính là: (i) quy trình dữ liệu có giữ được split dùng chung không, (ii)
CNN và ResNet18 đạt metric nào trên Test chung, và (iii) cặp lớp nào gây lỗi
nhiều nhất. Câu trả lời so sánh ResNet18 vẫn **PENDING** cho tới khi NV3 bàn giao
đủ artifact.

## PHẦN I. TỔNG QUAN

### 1.1. Bài toán phân loại bốn lớp

Đầu vào là ảnh màu của lá cà chua; đầu ra là một trong bốn nhãn cố định. Đây là
phân loại ảnh nhiều lớp, không phải object detection hay segmentation. Mapping
được đọc từ `configs/class_mapping.json`, không suy ra lại theo thứ tự thư mục.

### 1.2. CNN và Transfer Learning

CNN học bộ lọc đặc trưng trực tiếp từ ảnh thông qua các lớp tích chập và lớp
phân loại cuối. ResNet18 dùng kết nối dư để hỗ trợ tối ưu mạng sâu và được nhóm
khảo sát theo hướng pretrained backbone, feature extraction và fine-tuning.
Kiến trúc cụ thể, số tham số và hyperparameter chỉ được ghi sau khi đọc artifact
chính thức của NV2/NV3; không chép số minh họa từ mẫu báo cáo.

### 1.3. Ý nghĩa của đánh giá độc lập

Validation dùng để chọn cấu hình/checkpoint; Test chỉ dùng để báo cáo cuối cùng.
Đánh giá hai mô hình trên cùng sample ID và cùng ground truth giúp giảm sai lệch
do khác tập dữ liệu. Macro-F1 được báo cáo vì tính trung bình đều giữa các lớp,
không để lớp nhiều mẫu lấn át hoàn toàn lớp ít mẫu.

## PHẦN II. PHƯƠNG PHÁP THỰC HIỆN

### 2.1. Pipeline dữ liệu

Pipeline gồm manifest → mapping → transform → DataLoader → model → prediction
CSV → đánh giá NV5. Theo bàn giao NV1/PR #8, dataset có 6.502 ảnh: Train 4.115,
Validation 1.031 và Test 1.356. Train có thể augmentation; Validation/Test phải
dùng tiền xử lý tất định. **TODO:** nhóm xác nhận phiên bản manifest, thống kê
từng lớp, seed và bằng chứng loại trùng/leakage trong bản cuối.

### 2.2. CNN tự xây dựng

CNN do NV2 phụ trách, học từ đầu và chọn checkpoint bằng Validation. Artifact
PR #9 mô tả bốn khối Conv–BatchNorm–ReLU–MaxPool (số kênh 32 đến 256), Global
Average Pooling, Dropout 0,3 và lớp Linear bốn đầu ra. Cấu hình E1 có learning
rate 3e-4 được chọn bằng Validation; checkpoint epoch 24 đạt Validation
macro-F1 0,9768359521. NV5 không đổi code train hay chọn lại checkpoint, chỉ
kiểm tra prediction CSV, metrics JSON, mapping, split và confusion matrix đã
bàn giao. **TODO bản cuối:** chỉ điền optimizer/loss, batch size, thời gian và
số tham số khi metadata NV2 xác nhận đầy đủ.

### 2.3. ResNet18 Transfer Learning

ResNet18 do NV3 phụ trách, gồm feature extraction và fine-tuning theo Validation.
Tại HEAD hiện tại, PR #7 chưa có checkpoint fine-tuned, prediction CSV hoặc
metrics Test thật. Vì vậy mục kết quả ResNet18, số tham số, thời gian và confusion
matrix đều ghi **PENDING – chờ NV3 bàn giao**.

### 2.4. Quy tắc không dùng Test để chọn model

Checkpoint phải được chốt trước bằng `val_macro_f1`, tie-break bằng `val_loss`
nếu nhóm đã thống nhất. Script NV5 không train và không chỉnh hyperparameter;
nó chỉ tính/đối chiếu kết quả của artifact đã bàn giao.

## PHẦN III. THỰC NGHIỆM VÀ ĐÁNH GIÁ KẾT QUẢ

### 3.1. Giao thức và công cụ

Lệnh đánh giá từ metrics JSON hiện có:

```bash
python -m scripts.evaluate_models \
  --cnn-metrics results/cnn_metrics_nv2_handoff.json \
  --output-dir results
```

Khi có đủ prediction CSV:

```bash
python -m scripts.evaluate_models \
  --cnn-predictions results/cnn/cnn_test_predictions.csv \
  --resnet18-predictions results/resnet18/resnet18_test_predictions.csv \
  --output-dir results
```

Script kiểm tra schema cột, nhãn, split, sample ID, ground truth giữa hai model,
số mẫu metadata và consistency giữa metrics JSON với prediction CSV. Đầu ra là
`results/comparison.csv`, `results/analysis.md` và confusion matrix khi có
artifact tương ứng.

### 3.2. Kết quả CNN đã kiểm chứng

Artifact NV2 trong PR #9 ghi nhận 1.356 ảnh Test. NV5 đã chạy lại script với
prediction CSV thật và metrics JSON; các metric khớp trong sai số dưới `1e-6`:

| Chỉ số | CNN |
|---|---:|
| Accuracy | 0,969764 |
| Macro Precision | 0,965416 |
| Macro Recall | 0,964701 |
| Macro F1 | 0,965034 |
| Số ảnh Test | 1.356 |

F1 từng lớp: Healthy `0,990964`, Early Blight `0,927400`, Late Blight
`0,952128`, Bacterial Spot `0,989643`. Confusion matrix chi tiết và nguồn
metrics nằm trong `results/analysis.md` và `results/cnn_metrics_nv2_handoff.json`.

### 3.3. Phân tích lỗi CNN

CNN có 1.315 dự đoán đúng và 41 lỗi, tương đương khoảng 3,02%. Hai hướng nhầm
Early Blight → Late Blight (14) và Late Blight → Early Blight (12) cộng lại 26/41
lỗi, khoảng 63,4% tổng lỗi. Đây là cặp khó phân biệt nhất trong artifact Test
hiện có. Các lỗi còn lại gồm Late Blight → Healthy (5), Bacterial Spot → Late
Blight (4) và Early Blight → Bacterial Spot (3), cùng các cặp nhỏ hơn trong ma
trận.

### 3.4. Bảng so sánh hai mô hình

`results/comparison.csv` hiện có CNN ở trạng thái `computed`, ResNet18 ở trạng
thái `pending`. Không kết luận mô hình nào tốt hơn khi ResNet18 chưa có prediction
Test trên cùng sample ID. Số tham số và thời gian train/inference cũng để trống
nếu metadata chưa được chủ mô hình bàn giao.

### 3.5. Demo

NV4 chịu trách nhiệm mô tả UI Streamlit, API suy luận và ảnh minh họa. PR #1
đã ghi nhận CNN strict-load trên CPU và `demo.check_consistency` đạt
`max |delta| = 0` với checkpoint CNN thật; đó là bằng chứng luồng suy luận,
không thay thế metric Test. **TODO:** nhóm chèn ảnh/video đã được NV4 xác nhận,
lệnh chạy và checkpoint/hash thực tế dùng trong demo. Demo ResNet18 vẫn chờ
checkpoint fine-tuned từ NV3.

## PHẦN IV. KẾT LUẬN

Trong artifact đã kiểm chứng, CNN đạt Macro-F1 `0,965034` trên 1.356 ảnh Test.
Kết quả cho thấy nhóm lỗi chính nằm giữa Early Blight và Late Blight. Đây là
kết quả của một checkpoint đã được NV2 chọn trước bằng Validation, không phải
kết luận lựa chọn lại trên Test. Kết luận so sánh CNN–ResNet18 được để **PENDING**
cho đến khi NV3 bàn giao đầy đủ metrics, prediction CSV, checkpoint metadata và
confusion matrix.

### Hạn chế

- PlantVillage chủ yếu là ảnh trong điều kiện kiểm soát, không đại diện trực tiếp
  cho ảnh ngoài thực địa.
- Chỉ có bốn lớp; ảnh cây khác hoặc bệnh ngoài mapping vẫn bị ép vào một nhãn.
- Chưa có tập riêng cho bệnh mới chớm, nên không tuyên bố “phát hiện sớm”.
- Chưa có đủ artifact ResNet18 để so sánh công bằng.
- Chưa có gói ảnh đúng/sai đã được NV2 bàn giao để minh họa lỗi theo từng mẫu.

### Hướng phát triển

Nhận artifact ResNet18 và chạy lại cùng giao thức; bổ sung calibration, nhiều
seed, kiểm định ngoài miền và dữ liệu thực địa; kiểm thử demo end-to-end bằng
checkpoint thật; sau đó cập nhật đồng thời code, bảng, báo cáo và slide.

## TÀI LIỆU THAM KHẢO

1. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on
   plant health to enable the development of mobile disease diagnostics*.
   arXiv:1511.08060, https://arxiv.org/abs/1511.08060.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for
   Image Recognition*. CVPR, DOI: 10.1109/CVPR.2016.90,
   https://arxiv.org/abs/1512.03385.
3. scikit-learn developers. *precision_recall_fscore_support documentation*.
   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html.
4. PyTorch contributors. *torchvision.models.resnet18 documentation*.
   https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html.
5. Tài liệu nội bộ của nhóm: `docs/PHAN_CONG.md`,
   `docs/BAN_GIAO_NHIEM_VU_NHOM_HOC_MAY_CA_CHUA_23-10-2026.txt`, Issue #6,
   PR #7 và PR #9.

## PHỤ LỤC A. Bàn giao và truy vết

- Script NV5: `scripts/evaluate_models.py`.
- Bảng: `results/comparison.csv`.
- Phân tích: `results/analysis.md`.
- Metrics CNN handoff: `results/cnn_metrics_nv2_handoff.json`.
- Test NV5: `tests/test_nv5_evaluate_models.py`.
- **PENDING NV3:** `resnet18_metrics.json`, `resnet18_test_predictions.csv`,
  checkpoint/hash, split/seed, device, số tham số, thời gian và confusion matrix.
