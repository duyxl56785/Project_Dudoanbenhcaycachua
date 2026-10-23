# Phân công công việc

Nguồn: [tài liệu bàn giao 23/10/2026](BAN_GIAO_NHIEM_VU_NHOM_HOC_MAY_CA_CHUA_23-10-2026.txt). Checklist chi tiết của từng người nằm trong issue GitHub được gán cho người đó.

| NV  | Thành viên      | GitHub                                                              | Nhiệm vụ                          | Nhánh                  | Mục tài liệu |
|-----|-----------------|---------------------------------------------------------------------|-----------------------------------|------------------------|--------------|
| NV1 | Lê Xuân Bảo     | [@lexuanbao258-collab](https://github.com/lexuanbao258-collab)       | Dataset & Preprocessing           | `feature/nv1-dataset`  | III          |
| NV2 | Nguyễn Văn Duy  | [@duyxl56785](https://github.com/duyxl56785)                         | CNN tự xây dựng                   | `feature/nv2-cnn`      | IV           |
| NV3 | Trần Văn Chính  | [@tranvanchinhbg2006-alt](https://github.com/tranvanchinhbg2006-alt) | ResNet18 & Transfer Learning      | `feature/nv3-resnet18` | V            |
| NV4 | Nguyễn Minh Duy | [@nguyenminhduy090-alt](https://github.com/nguyenminhduy090-alt)     | Demo Streamlit & tích hợp mô hình | `feature/nv4-demo`     | VI           |
| NV5 | Phạm Thế Cường  | [@phamthecuong2022006-alt](https://github.com/phamthecuong2022006-alt) | Đánh giá, báo cáo, slide        | `feature/nv5-report`   | VII          |

## NV1 – Lê Xuân Bảo (@lexuanbao258-collab)

- **Nhiệm vụ:** tạo bộ dữ liệu 4 lớp sạch, rõ nguồn gốc; chia Train/Val/Test chống rò rỉ; pipeline tiền xử lý và augmentation dùng chung.
- **Gate:** chủ trì Gate 0 và Gate 1 (ưu tiên số 1, mở đường cho NV2/NV3).
- **File phụ trách:** `configs/class_mapping.json`, `data/`, `src/data/`, `scripts/prepare_dataset.py`, `scripts/check_dataset.py`, `scripts/visualize_dataset.py`, `docs/DATASET_README.md`, `figures/class_distribution.png`, `figures/augmentation_examples.png`.
- **Bàn giao cho:** NV2, NV3 (split, transforms); NV4 (mapping, transforms); NV5 (thống kê, biểu đồ, cách chia tập).
- **Báo cáo:** Mở đầu, Phần I (nội dung nền), Phần II (pipeline dữ liệu), Phần III (dữ liệu, chia tập, augmentation).

## NV2 – Nguyễn Văn Duy (@duyxl56785)

- **Nhiệm vụ:** thiết kế và huấn luyện CNN từ đầu (không pretrained), 4 lớp; chọn cấu hình chỉ bằng Validation.
- **Gate:** Gate 2 (song song NV3), sau khi NV1 đạt Gate 1.
- **File phụ trách:** `src/models/cnn.py`, `scripts/train_cnn.py`, `scripts/eval_cnn.py`, `scripts/infer_cnn.py`, `configs/cnn.yaml`, `checkpoints/cnn_best.pt`, `results/cnn/`, `figures/cnn_train_curves.png`, `docs/CNN_README.md`.
- **Bàn giao cho:** NV4 (model definition, checkpoint, preprocessing, API predict); NV5 (log, cấu hình, metrics, confusion matrix, test predictions).
- **Báo cáo:** Phần II (CNN), Phần III (cấu hình/đồ thị/kết quả CNN).

## NV3 – Trần Văn Chính (@tranvanchinhbg2006-alt)

- **Nhiệm vụ:** ResNet18 pretrained, so sánh feature extraction với fine-tuning; chọn cấu hình chỉ bằng Validation.
- **Gate:** Gate 2 (song song NV2), sau khi NV1 đạt Gate 1.
- **File phụ trách:** `src/models/resnet18.py`, `scripts/train_resnet18.py`, `scripts/eval_resnet18.py`, `scripts/infer_resnet18.py`, `configs/resnet18.yaml`, `checkpoints/resnet18_best.pt`, `results/resnet18/`, `figures/resnet18_train_curves.png`, `docs/RESNET18_README.md`.
- **Bàn giao cho:** NV4 (checkpoint, inference API, preprocessing, mapping); NV5 (log, metrics, tham số, test predictions).
- **Báo cáo:** Phần II (ResNet18 & Transfer Learning), Phần III (cấu hình/đồ thị/kết quả ResNet18).

## NV4 – Nguyễn Minh Duy (@nguyenminhduy090-alt)

- **Nhiệm vụ:** demo Streamlit upload ảnh, dự đoán bằng CNN/ResNet18 hoặc so sánh cả hai; dùng mô hình thật trước nghiệm thu.
- **Gate:** làm UI mock trong Gate 2; chủ trì Gate 4 (tích hợp checkpoint thật).
- **File phụ trách:** `src/inference.py`, `demo/`, `requirements.txt`, `docs/DEMO_README.md`.
- **Bàn giao cho:** NV5 (video/ảnh và hướng dẫn chạy demo).
- **Báo cáo:** Phần III (mô tả demo, ảnh minh họa, cách dùng).

## NV5 – Phạm Thế Cường (@phamthecuong2022006-alt)

- **Nhiệm vụ:** kiểm định độc lập và so sánh hai mô hình trên Test chung; tổng hợp báo cáo, slide, chuẩn bị bảo vệ. Tổng hợp và kiểm tra, không làm thay phần người khác.
- **Gate:** lập khung báo cáo từ Gate 0; chủ trì Gate 3 và Gate 5.
- **File phụ trách:** `scripts/evaluate_models.py`, `results/comparison.csv`, `results/analysis.md`, `report/`.
- **Báo cáo:** ghép và chuẩn hóa toàn bộ; Phần III (giao thức đánh giá, bảng so sánh, phân tích); Phần IV (kết luận); tài liệu tham khảo; phụ lục.

## Quy tắc

- Mỗi người làm trên nhánh của mình, mở Pull Request vào `main`, điền checklist bàn giao (a)–(f) và có người review trước khi merge.
- Không sửa file của người khác khi chưa trao đổi. Mapping, split và quy ước đánh giá là cấu hình chung: thay đổi phải được NV1 thông báo và cả nhóm đồng thuận.
