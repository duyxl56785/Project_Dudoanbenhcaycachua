# Nhận diện bệnh qua ảnh lá cà chua bằng CNN và ResNet18

Đồ án nhóm học phần **Học máy và Khai phá dữ liệu** – Trường Đại học CMC.

Đề tài: *Ứng dụng CNN và Transfer Learning trong nhận diện bệnh trên lá cà chua qua hình ảnh.*

Phạm vi kỹ thuật đã chốt: **phân loại ảnh lá cà chua theo 4 lớp** (không khoanh vùng bệnh), so sánh hai hướng:

1. CNN tự xây dựng, huấn luyện từ đầu (không pretrained).
2. ResNet18 pretrained, Transfer Learning (feature extraction rồi fine-tuning).

Kèm demo Streamlit để tải ảnh lá và xem dự đoán. Công nghệ: Python, PyTorch/torchvision, Streamlit. Dữ liệu: PlantVillage (phiên bản/link sẽ ghi trong [docs/DATASET_README.md](docs/DATASET_README.md)).

> Trạng thái: **mới khởi tạo khung dự án**. Chưa có dữ liệu, mô hình hay kết quả thực nghiệm.

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
├── requirements.txt
├── .gitignore
├── .github/
│   └── pull_request_template.md   # checklist bàn giao (a)-(f)
├── configs/
│   ├── class_mapping.json         # ánh xạ nhãn dùng chung
│   ├── cnn.yaml                   # NV2
│   └── resnet18.yaml              # NV3
├── data/
│   ├── README.md                  # cách tải/kiểm tra dữ liệu (không commit ảnh)
│   └── splits/                    # train.csv, val.csv, test.csv (NV1)
├── src/
│   ├── data/
│   │   ├── dataset.py             # NV1
│   │   └── transforms.py          # NV1
│   ├── models/
│   │   ├── cnn.py                 # NV2
│   │   └── resnet18.py            # NV3
│   └── inference.py               # NV4
├── scripts/
│   ├── prepare_dataset.py         # NV1
│   ├── check_dataset.py           # NV1
│   ├── visualize_dataset.py       # NV1
│   ├── train_cnn.py               # NV2
│   ├── eval_cnn.py                # NV2
│   ├── infer_cnn.py               # NV2
│   ├── train_resnet18.py          # NV3
│   ├── eval_resnet18.py           # NV3
│   ├── infer_resnet18.py          # NV3
│   └── evaluate_models.py         # NV5
├── checkpoints/
│   └── README.md                  # link trọng số nếu tách khỏi Git
├── results/
│   ├── cnn/
│   └── resnet18/
├── figures/
├── demo/
│   ├── app.py                     # NV4
│   └── screenshots/
├── report/                        # báo cáo .docx, slide (NV5)
└── docs/
    ├── BAN_GIAO_NHIEM_VU_NHOM_HOC_MAY_CA_CHUA_23-10-2026.txt   # tài liệu phân công
    ├── PHAN_CONG.md               # bảng phân công kèm tài khoản GitHub
    ├── DATASET_README.md
    ├── CNN_README.md
    ├── RESNET18_README.md
    ├── DEMO_README.md
    └── MEETING_NOTES.md
```

## Cài đặt

Yêu cầu: Python 3 (phiên bản cụ thể sẽ ghi khi chốt môi trường thực nghiệm).

```bash
git clone https://github.com/duyxl56785/Project_Dudoanbenhcaycachua.git
cd Project_Dudoanbenhcaycachua
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Nếu cần PyTorch bản GPU (CUDA), cài `torch`/`torchvision` theo hướng dẫn tại https://pytorch.org trước khi cài phần còn lại. Trên Google Colab, ghi lại phiên bản thư viện thực tế đã dùng.

Hướng dẫn tải dữ liệu: [data/README.md](data/README.md). Hướng dẫn chạy từng phần: xem các file trong [docs/](docs/).

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

Chưa có ngày hạn chính thức; nhóm tự điền theo lịch giảng viên.

| Gate | Nội dung                                   | Chủ trì            | Hạn         | Trạng thái |
|------|--------------------------------------------|--------------------|-------------|------------|
| 0    | Thống nhất kho chung, quy tắc, mapping     | NV1, cả nhóm       | (chưa điền) | Chưa làm   |
| 1    | Chốt dataset, split, transforms            | NV1                | (chưa điền) | Chưa làm   |
| 2    | Có mô hình huấn luyện (CNN + ResNet18)     | NV2, NV3           | (chưa điền) | Chưa làm   |
| 3    | Kiểm định thực nghiệm trên Test chung      | NV5                | (chưa điền) | Chưa làm   |
| 4    | Demo tích hợp mô hình thật                 | NV4                | (chưa điền) | Chưa làm   |
| 5    | Đóng gói nộp và bảo vệ                     | Cả nhóm            | (chưa điền) | Chưa làm   |
