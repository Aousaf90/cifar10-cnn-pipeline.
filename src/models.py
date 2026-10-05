import torch.nn as nn


def build_model(filters, dropout):
    # Conv -> BatchNorm -> ReLU -> MaxPool -> Conv -> BatchNorm -> ReLU -> MaxPool
    # -> Dense -> Dropout -> Output(10)
    return nn.Sequential(
        nn.Conv2d(3, filters, 3, padding=1), nn.BatchNorm2d(filters), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(filters, 2 * filters, 3, padding=1), nn.BatchNorm2d(2 * filters), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(2 * filters * 8 * 8, 256), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 10),
    )