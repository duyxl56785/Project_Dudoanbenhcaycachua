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
nhiều nhất. NV5 đã kiểm tra hai prediction CSV cùng 1.356 sample ID sau khi
chuẩn hóa tiền tố path Colab/Windows; kết quả so sánh dưới đây dùng artifact đã
được tải, kiểm hash checkpoint và tính lại metric.

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
Gói bàn giao PR #7 có checkpoint fine-tuned, metadata, CSV Test, metrics,
confusion matrix, log và training curve. NV5 tải ZIP, kiểm SHA-256 checkpoint
`61d44ba8ac02ee57c44cbd3835aaac2c657c97924735fea743a60df5f5fdf9e2`, rồi
đối chiếu hai CSV trên cùng Test. Checkpoint selected stage `fine_tuning`, epoch
9, Validation macro-F1 `0,980859`; ResNet18 có 11.178.564 tham số, trong đó
8.395.780 tham số trainable ở stage chọn. Artifact không bàn giao thời gian train
hoặc inference nên các trường đó để trống.

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

### 3.2. Kết quả Test đã kiểm chứng

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

| Chỉ số | ResNet18 |
|---|---:|
| Accuracy | 0,986726 |
| Macro Precision | 0,985348 |
| Macro Recall | 0,982732 |
| Macro F1 | 0,983978 |
| Số ảnh Test | 1.356 |

F1 ResNet18 theo lớp: Healthy `0,993958`, Early Blight `0,962264`, Late Blight
`0,985430`, Bacterial Spot `0,994259`. Đây là kết quả NV5 tính lại từ prediction
CSV của gói NV3 và đối chiếu metrics JSON sau khi adapter chuyển
`classification_report` sang schema NV5; không phải số chép tay từ PR.

### 3.3. Phân tích lỗi CNN

CNN có 1.315 dự đoán đúng và 41 lỗi, tương đương khoảng 3,02%. Hai hướng nhầm
Early Blight → Late Blight (14) và Late Blight → Early Blight (12) cộng lại 26/41
lỗi, khoảng 63,4% tổng lỗi. Đây là cặp khó phân biệt nhất trong artifact Test
hiện có. Các lỗi còn lại gồm Late Blight → Healthy (5), Bacterial Spot → Late
Blight (4) và Early Blight → Bacterial Spot (3), cùng các cặp nhỏ hơn trong ma
trận.

ResNet18 đúng 1.338/1.356 ảnh và sai 18 ảnh. Cặp nhầm nhiều nhất vẫn là Early
Blight → Late Blight (6 ảnh); Early Blight → Healthy và Late Blight → Early
Blight có 3 ảnh mỗi hướng. So với CNN, ResNet18 cao hơn `0,016962` Accuracy và
`0,018944` macro-F1 trên cùng Test. ImageNet pretraining rồi fine-tuning là giải
thích kỹ thuật hợp lý cho chênh lệch, nhưng không chứng minh quan hệ nhân quả;
cần nhiều seed và dữ liệu thực địa để khẳng định độ ổn định.

### 3.4. Bảng so sánh hai mô hình

`results/comparison.csv` hiện có hai dòng `computed`. CNN có 389.924 tham số,
1.118,7 giây train trong run E1 trên RTX 3050 theo run_info NV2; ResNet18 có
11.178.564 tổng tham số và 8.395.780 tham số trainable theo artifact NV3. Thời
gian train/inference ResNet18 không được bàn giao nên để trống. Các nguồn, hash
checkpoint, phiên bản môi trường và Validation score nằm trong hai JSON handoff.

### 3.5. Demo

NV4 chịu trách nhiệm mô tả UI Streamlit, API suy luận và ảnh minh họa. PR #1
đã ghi nhận CNN strict-load trên CPU và `demo.check_consistency` đạt
`max |delta| = 0` với checkpoint CNN thật; đó là bằng chứng luồng suy luận,
không thay thế metric Test. **TODO:** nhóm chèn ảnh/video đã được NV4 xác nhận,
lệnh chạy và checkpoint/hash thực tế dùng trong demo. Sau bàn giao NV3, PR #1
ghi nhận ResNet18 strict-load CPU, `demo.check_consistency --model both` có
`max |delta| = 0` và tái hiện metric Test; ảnh/video minh họa cuối vẫn cần NV4
xác nhận trước khi nộp.

## PHẦN IV. KẾT LUẬN

Trong artifact đã kiểm chứng, ResNet18 đạt Macro-F1 `0,983978` trên 1.356 ảnh
Test, cao hơn CNN `0,965034` trên cùng sample ID/ground truth. Cả hai checkpoint
được chủ mô hình chọn bằng Validation trước khi Test; NV5 không dùng Test để chọn
lại. Lỗi của hai model vẫn tập trung ở Early Blight/Late Blight. Kết luận này chỉ
áp dụng cho artifact PlantVillage hiện có.

### Hạn chế

- PlantVillage chủ yếu là ảnh trong điều kiện kiểm soát, không đại diện trực tiếp
  cho ảnh ngoài thực địa.
- Chỉ có bốn lớp; ảnh cây khác hoặc bệnh ngoài mapping vẫn bị ép vào một nhãn.
- Chưa có tập riêng cho bệnh mới chớm, nên không tuyên bố “phát hiện sớm”.
- Chưa có thí nghiệm nhiều seed hoặc kiểm định ngoài miền để khẳng định chênh
  lệch CNN–ResNet18 ổn định.
- Chưa có gói ảnh lỗi CNN có nguồn truy vết nằm trong repository NV5 để chèn vào
  báo cáo; chỉ thêm ảnh khi NV2 bàn giao file/link chính thức.

### Hướng phát triển

Bổ sung calibration, nhiều seed, kiểm định ngoài miền và dữ liệu thực địa; hoàn
tất ảnh/video demo end-to-end bằng checkpoint thật; sau đó nhóm trưởng kiểm tra
lại bảng, báo cáo và slide trước khi nộp.

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
- Metrics ResNet18 handoff: `results/resnet18_metrics_nv3_handoff.json`.
- Test NV5: `tests/test_nv5_evaluate_models.py`.
- ZIP nguồn và checkpoint nhị phân ResNet18 không được commit vào branch NV5;
  link nguồn và hash checkpoint được ghi trong JSON handoff.
