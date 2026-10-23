# Dữ liệu

Phụ trách: NV1 - Lê Xuân Bảo.

Ảnh **không commit lên Git**. `data/raw/` và `data/processed/` đã nằm trong `.gitignore`. Chỉ commit manifest split và bảng thống kê.

## Nguồn

- Dataset: PlantVillage
- Link: (chưa điền)
- Phiên bản: (chưa điền)
- Ngày tải: (chưa điền)
- Giấy phép: (chưa điền)

## Cách tải và đặt dữ liệu

(chưa điền) – lệnh/các bước tải, nơi giải nén (gợi ý `data/raw/`).

## Cấu trúc dự kiến

```
data/
├── raw/                  # ảnh gốc tải về (không commit)
├── processed/            # ảnh đã xử lý nếu có (không commit)
├── splits/
│   ├── train.csv         # path + label + group (nếu có)
│   ├── val.csv
│   └── test.csv
└── dataset_summary.csv   # thống kê lớp, số ảnh theo tập
```

## Kiểm tra dữ liệu

(chưa điền) – lệnh chạy `scripts/prepare_dataset.py`, `scripts/check_dataset.py`, `scripts/visualize_dataset.py`.

Chi tiết quy trình, thống kê và hạn chế: [docs/DATASET_README.md](../docs/DATASET_README.md).
