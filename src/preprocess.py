import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from data_io import load_split, save_split

p = yaml.safe_load(open("params.yaml"))["preprocess"]

x, y = load_split("data/raw", "train")
xt, yt = load_split("data/raw", "test")
assert x.shape[1] == p["image_size"], "image_size in params.yaml does not match data"

x = x.astype("float32") / 255.0
xt = xt.astype("float32") / 255.0

xtr, xv, ytr, yv = train_test_split(
    x, y, test_size=p["val_size"], random_state=p["seed"], stratify=y
)

# Normalization (per-channel standardization using TRAIN statistics only)
mean, std = xtr.mean(axis=(0, 1, 2)), xtr.std(axis=(0, 1, 2))
norm = lambda a: ((a - mean) / std).transpose(0, 3, 1, 2).astype("float16")  # NHWC -> NCHW

# processed train (~276 MB float16) -> 10 shards; val -> 1; test -> 2
save_split("data/processed", "train", norm(xtr), ytr, 10)
save_split("data/processed", "val", norm(xv), yv, 1)
save_split("data/processed", "test", norm(xt), yt, 2)