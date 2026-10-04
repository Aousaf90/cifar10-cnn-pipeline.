import os, numpy as np
from torchvision import datasets

os.makedirs("data/raw", exist_ok=True)
for name, train in [("train", True), ("test", False)]:
    ds = datasets.CIFAR10("data/_download", train=train, download=True)
    np.savez(f"data/raw/{name}.npz", x=ds.data, y=np.array(ds.targets))