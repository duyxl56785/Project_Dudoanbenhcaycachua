"""Mô hình ResNet18 pretrained cho Transfer Learning.

Phụ trách: NV3 - Trần Văn Chính.

Việc cần làm (mục V tài liệu bàn giao):
- Load ResNet18 bằng torchvision với pretrained weights; thay
  classification head để ra 4 lớp.
- Hỗ trợ Bước A: đóng băng backbone, chỉ huấn luyện head
  (feature extraction).
- Hỗ trợ Bước B: fine-tune một phần/toàn bộ backbone; có thể đặt learning
  rate backbone nhỏ hơn head.
- Bàn giao model definition cho NV4 (demo) và NV5 (đánh giá).
"""
