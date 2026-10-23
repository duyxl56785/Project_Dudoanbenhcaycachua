"""Huấn luyện ResNet18 bằng Transfer Learning.

Phụ trách: NV3 - Trần Văn Chính.

Việc cần làm (mục V tài liệu bàn giao):
- Đọc configs/resnet18.yaml; dùng split và class_mapping chung của NV1.
- Bước A: feature extraction (đóng băng backbone, huấn luyện head).
- Bước B: fine-tuning một phần/toàn bộ backbone.
- Theo dõi train/val loss, accuracy; chọn checkpoint theo Validation,
  lưu checkpoints/resnet18_best.pt.
- Xuất results/resnet18_feature_vs_finetune.csv và
  figures/resnet18_train_curves.png.
"""
