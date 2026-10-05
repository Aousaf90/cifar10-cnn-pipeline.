import numpy as np
from torchvision import datasets
from data_io import save_split

# raw train (~154 MB) -> 5 shards of ~31 MB; raw test (~31 MB) -> 1 shard
for name, train, shards in [("train", True, 5), ("test", False, 1)]:
    ds = datasets.CIFAR10("data/_download", train=train, download=True)
    save_split("data/raw", name, ds.data, np.array(ds.targets), shards)