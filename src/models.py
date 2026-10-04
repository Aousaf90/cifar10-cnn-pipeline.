import torch.nn as nn

def build_model(f, dropout):
    return nn.Sequential(
        nn.Conv2d(3, f, 3, padding=1), nn.BatchNorm2d(f), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(f, 2 * f, 3, padding=1), nn.BatchNorm2d(2 * f), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(2 * f * 8 * 8, 256), nn.ReLU(),
        nn.Dropout(dropout), nn.Linear(256, 10))