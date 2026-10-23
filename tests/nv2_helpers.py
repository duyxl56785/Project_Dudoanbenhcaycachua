"""Tiện ích chỉ dùng cho test của NV2. Dữ liệu ở đây là tensor ngẫu nhiên,
KHÔNG phải ảnh lá cà chua và không dùng để báo cáo kết quả."""
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

NAMES = ["Healthy", "Early Blight", "Late Blight", "Bacterial Spot"]


def synthetic_loader(n, seed=0, shuffle=True):
    gen = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, 224, 224, generator=gen)
    labels = torch.arange(n) % 4
    return DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def load_yaml(path):
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def simple_transform(size):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
