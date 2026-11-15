# Nhận diện bệnh qua ảnh lá cà chua bằng CNN và ResNet18

Đồ án nhóm học phần **Học máy và Khai phá dữ liệu** – Trường Đại học CMC.

Đề tài: *Ứng dụng CNN và Transfer Learning trong nhận diện bệnh trên lá cà chua qua hình ảnh.*

Phạm vi kỹ thuật đã chốt: **phân loại ảnh lá cà chua theo 4 lớp** (không khoanh vùng bệnh), so sánh hai hướng:

1. CNN tự xây dựng, huấn luyện từ đầu (không pretrained).
2. ResNet18 pretrained, Transfer Learning (feature extraction rồi fine-tuning).

Kèm demo Streamlit để tải ảnh lá và xem dự đoán (có Grad-CAM thử nghiệm). Ngoài hai mô hình chính, nhóm bổ sung hai baseline học máy truyền thống (HOG + SVM) để so sánh. Công nghệ: Python, PyTorch/torchvision, scikit-learn, scikit-image, Streamlit. Dữ liệu: 6.502 ảnh màu từ [PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage), xem [docs/DATASET_README.md](docs/DATASET_README.md).

> Trạng thái (15/11/2026): **đã tích hợp dữ liệu, hai mô hình chính, hai baseline SVM, demo và đánh giá vào `main`.** Còn lại khâu đóng gói nộp: video demo, bản báo cáo/slide cuối và các mục chưa đóng ở [Tiến độ](#tiến-độ).

## Kết quả trên tập Test

Tập Test cố định **1.356 ảnh**; mọi mô hình chọn checkpoint/cấu hình bằng **Validation macro-F1**, Test chỉ dùng để báo cáo. Nguồn: [`results/comparison.csv`](results/comparison.csv).

| Mô hình | Accuracy | Macro-F1 | F1 Early Blight | Ghi chú |
|---|---:|---:|---:|---|
| ResNet18 (transfer learning) | **98,67%** | **0,9840** | 0,9623 | Mô hình chính |
| CNN tự xây dựng | 96,98% | 0,9650 | 0,9274 | Mô hình chính, 389.924 tham số |
| HOG + HSV + RBF-SVM | 92,85% | 0,9096 | 0,7789 | Baseline NV5 |
| HOG + Linear SVM V1 | 73,89% | 0,6940 | 0,3777 | Baseline NV4 |

Early Blight ↔ Late Blight là cặp nhầm lẫn khó nhất ở mọi mô hình. Chênh lệch ResNet18–CNN đo trên một seed và một tập Test trong miền PlantVillage, chưa kiểm định ngoài miền. Phân tích chi tiết: [`results/analysis.md`](results/analysis.md).

📄 **Mọi thành viên đọc trước:** [Tài liệu bàn giao nhiệm vụ (23/10/2026)](docs/BAN_GIAO_NHIEM_VU_NHOM_HOC_MAY_CA_CHUA_23-10-2026.txt) – phân công, hợp đồng kỹ thuật dùng chung, việc phải làm, tệp bàn giao và điều kiện nghiệm thu của từng NV, các Gate.

## Các lớp

Ánh xạ nhãn cố định trong [configs/class_mapping.json](configs/class_mapping.json). Mọi mô hình, script đánh giá và demo phải đọc chung file này; không dùng thứ tự alphabet mặc định của `ImageFolder`.

| Nhãn | Lớp            | Mô tả                      |
|------|----------------|----------------------------|
| 0    | Healthy        | Lá khỏe mạnh               |
| 1    | Early Blight   | Bệnh đốm vòng/cháy lá sớm  |
| 2    | Late Blight    | Bệnh mốc sương             |
| 3    | Bacterial Spot | Bệnh đốm vi khuẩn          |

## Phân công

| NV  | Thành viên      | GitHub                     | Nhiệm vụ                                   | Nhánh gợi ý            | Tài liệu |
|-----|-----------------|----------------------------|--------------------------------------------|------------------------|----------|
| NV1 | Lê Xuân Bảo     | @lexuanbao258-collab       | Dataset & Preprocessing                    | `feature/nv1-dataset`  | [DATASET_README](docs/DATASET_README.md) |
| NV2 | Nguyễn Văn Duy  | @duyxl56785                | CNN tự xây dựng                            | `feature/nv2-cnn`      | [CNN_README](docs/CNN_README.md) |
| NV3 | Trần Văn Chính  | @tranvanchinhbg2006-alt    | ResNet18 & Transfer Learning               | `feature/nv3-resnet18` | [RESNET18_README](docs/RESNET18_README.md) |
| NV4 | Nguyễn Minh Duy | @nguyenminhduy090-alt      | Demo Streamlit & tích hợp mô hình          | `feature/nv4-demo`     | [DEMO_README](docs/DEMO_README.md) |
| NV5 | Phạm Thế Cường  | @phamthecuong2022006-alt   | Kiểm thử, so sánh, báo cáo, slide          | `feature/nv5-report`   | `results/`, `report/` |

Mỗi người chịu trách nhiệm chính phần của mình và cung cấp số liệu, hình, giải thích cho phần đó. NV5 tổng hợp và kiểm tra, không làm thay phần của người khác.

Chi tiết file phụ trách, Gate và người nhận bàn giao của từng người: [docs/PHAN_CONG.md](docs/PHAN_CONG.md). Checklist công việc nằm trong issue GitHub gán cho từng người.

## Cấu trúc thư mục

```
Project_Dudoanbenhcaycachua/
├── README.md
├── requirements.txt               # demo, CNN, ResNet18, HOG-SVM V1
├── requirements-*.txt             # môi trường riêng cho HOG-SVM / SVM NV5 / test
├── .github/workflows/             # CI: demo, HOG-SVM, SVM NV5
├── configs/                       # class_mapping.json (chung), cnn.yaml, resnet18.yaml
├── data/
│   ├── README.md                  # cách tải ZIP dataset từ Drive (không commit ảnh)
│   ├── dataset_manifest.csv, audit/
│   └── splits/                    # train.csv, val.csv, test.csv (NV1)
├── src/
│   ├── data/                      # dataset.py, transforms.py (NV1)
│   ├── models/                    # cnn.py (NV2), resnet18.py (NV3)
│   ├── inference.py               # nạp checkpoint cho demo (NV4)
│   ├── explainability.py          # Grad-CAM
│   └── hog_svm.py                 # đặc trưng HOG/HSV cho SVM
├── scripts/                       # prepare/check dataset, train/eval/infer từng mô hình, evaluate_models.py
├── checkpoints/                   # cnn_best.pt trong Git; resnet18_best.pt tải từ Drive
├── results/                       # metrics, prediction CSV, confusion matrix, comparison.csv
├── figures/
├── demo/                          # app.py (Streamlit), check_consistency.py, screenshots/
├── report/                        # báo cáo, slide, checklist nghiệm thu (NV5)
├── tests/                         # pytest/unittest cho từng NV
└── docs/                          # README từng phần, audit, Q&A bảo vệ
```

## Cài đặt

Yêu cầu: **Python 3.12** (môi trường CI và môi trường huấn luyện CNN dùng 3.12).

```bash
git clone https://github.com/duyxl56785/Project_Dudoanbenhcaycachua.git
cd Project_Dudoanbenhcaycachua
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Nếu cần PyTorch bản GPU (CUDA), cài `torch`/`torchvision` theo hướng dẫn tại https://pytorch.org trước khi cài phần còn lại. Trên Google Colab, ghi lại phiên bản thư viện thực tế đã dùng.

### Checkpoint

| File | Có trong Git? | Cách lấy | SHA-256 |
|---|---|---|---|
| `checkpoints/cnn_best.pt` | Có | – | `d33c95bbe15e82fa83f1fd954136ebfff6e1cbba1a86fe40af473a377390de1f` |
| `checkpoints/resnet18_best.pt` (44,8 MB) | Không | Giải nén [gói NV3 trên Drive](https://drive.google.com/file/d/153DOn8zLCiIhnydLoPNDcht3X0EFhKuR/view), đặt `resnet18_best.pt` và `resnet18_best.meta.json` vào `checkpoints/` | `61d44ba8ac02ee57c44cbd3835aaac2c657c97924735fea743a60df5f5fdf9e2` |
| `results/hog_svm_nv5_experiment/hog_hsv_rbf_svm.joblib` (25 MB) | Không | Xem [CHECKPOINT_HANDOFF.md](results/hog_svm_nv5_experiment/CHECKPOINT_HANDOFF.md); cần scikit-learn 1.6.1 | `d40cf8af63242ebfe8e149eba333419de2c2551999589b7df2d5710db21cfcd5` |

Kiểm tra hash trước khi nạp: `Get-FileHash -Algorithm SHA256 <file>` (PowerShell) hoặc `sha256sum <file>`. Chi tiết: [checkpoints/README.md](checkpoints/README.md).

### Chạy nhanh

```bash
streamlit run demo/app.py                              # demo (cần checkpoint ở trên)
python -m demo.check_consistency <anh.jpg> --model both   # đối chiếu demo với script suy luận
python -m pytest tests -q                              # test; phần cần ảnh/checkpoint sẽ tự skip
```

Hướng dẫn tải dữ liệu: [data/README.md](data/README.md). Hướng dẫn chạy từng phần: [CNN](docs/CNN_README.md), [ResNet18](docs/RESNET18_README.md), [Demo](docs/DEMO_README.md), [đánh giá NV5](docs/NV5_EVALUATION.md).

## Quy tắc chung

1. **Dùng chung `configs/class_mapping.json` và split** do NV1 chốt (`data/splits/`). Không tự chia lại dữ liệu. Thay đổi mapping/split phải được NV1 thông báo và cả nhóm đồng thuận; sau Gate 1, sửa split thì tạo phiên bản mới và train/đánh giá lại.
2. **Augmentation chỉ áp dụng cho Train.** Validation/Test dùng tiền xử lý tất định.
3. **Không dùng Test để chọn model**, chọn checkpoint hay chỉnh hyperparameter. Chỉ chọn bằng Validation.
4. **Không số liệu giả.** Mọi Accuracy/F1/số ảnh/thời gian phải truy vết được đến log, checkpoint, script.
5. **Mỗi người một nhánh** (`feature/nvX-...`), mở Pull Request vào `main` và **có người review trước khi merge**. Điền đủ checklist bàn giao trong PR template.
6. Không sửa file của người khác khi chưa trao đổi. Không commit API key, mật khẩu, dữ liệu nhạy cảm, dataset lớn.

## Hạn chế cần thừa nhận

- PlantVillage chủ yếu là ảnh lá trong điều kiện được kiểm soát; hiệu quả ngoài thực địa có thể thấp hơn.
- Mô hình là công cụ hỗ trợ phân loại ảnh, **không phải chẩn đoán chắc chắn**; xác suất softmax không mặc nhiên đã hiệu chỉnh.
- Chỉ có 4 lớp: ảnh bệnh khác hoặc cây khác vẫn bị ép dự đoán vào một trong 4 lớp.
- Chưa có đánh giá riêng trên ảnh bệnh mới chớm, nên **không tuyên bố "phát hiện sớm"** đã được kiểm chứng.

## Tiến độ

Chưa có ngày hạn chính thức; nhóm tự điền theo lịch giảng viên. Trạng thái cập nhật 15/11/2026 theo bằng chứng trong repo.

| Gate | Nội dung                                   | Chủ trì            | Hạn         | Trạng thái |
|------|--------------------------------------------|--------------------|-------------|------------|
| 0    | Thống nhất kho chung, quy tắc, mapping     | NV1, cả nhóm       | (chưa điền) | Xong – `configs/class_mapping.json`, PR template, CI |
| 1    | Chốt dataset, split, transforms            | NV1                | (chưa điền) | Xong phần kỹ thuật (PR #8 đã merge; NV4 xác minh SHA-256 đủ 6.502 ảnh). Còn mục chờ xác nhận ở [DATASET_README §9](docs/DATASET_README.md#9-trạng-thái-bàn-giao) |
| 2    | Có mô hình huấn luyện (CNN + ResNet18)     | NV2, NV3           | (chưa điền) | Xong – PR #9, PR #7 |
| 3    | Kiểm định thực nghiệm trên Test chung      | NV5                | (chưa điền) | Xong – [`results/comparison.csv`](results/comparison.csv), PR #18 |
| 4    | Demo tích hợp mô hình thật                 | NV4                | (chưa điền) | Xong – [audit 14/11](docs/audits/NV4_FINAL_AUDIT_20261114.md); chưa quay video |
| 5    | Đóng gói nộp và bảo vệ                     | Cả nhóm            | (chưa điền) | Đang làm – video demo, báo cáo/slide bản cuối |
