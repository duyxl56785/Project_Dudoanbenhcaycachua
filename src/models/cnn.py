"""Mô hình CNN tự xây dựng, huấn luyện từ đầu (không pretrained weights).

Phụ trách: NV2 - Nguyễn Văn Duy.

Kiến trúc (giải thích chi tiết trong docs/CNN_README.md):
4 khối Conv3x3 -> BatchNorm -> ReLU -> MaxPool2, số kênh 32-64-128-256,
rồi Global Average Pooling -> Dropout -> Linear ra 4 lớp.
Với ảnh 128x128: 3x128x128 -> 32x64x64 -> 64x32x32 -> 128x16x16 -> 256x8x8 -> 256 -> 4.

Hợp đồng với demo NV4 (src/inference.py): build_model(num_classes=4) và
checkpoint là state_dict thuần của mô hình này.
"""
from torch import nn

NUM_CLASSES = 4
CHANNELS = (32, 64, 128, 256)
DROPOUT = 0.3


def _conv_block(in_channels, out_channels):
    # Conv không bias vì BatchNorm phía sau đã có tham số dịch (beta).
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class TomatoCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        blocks, in_channels = [], 3
        for out_channels in CHANNELS:
            blocks.append(_conv_block(in_channels, out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(self.dropout(x))


def build_model(num_classes=NUM_CLASSES):
    """Tạo CNN mới, khởi tạo ngẫu nhiên. Chỉ hỗ trợ đúng 4 lớp của đề tài."""
    if num_classes != NUM_CLASSES:
        raise ValueError(f"CNN của đề tài cần đúng {NUM_CLASSES} lớp, nhận {num_classes}")
    return TomatoCNN(num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
