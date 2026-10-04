import os, yaml, numpy as np
from sklearn.model_selection import train_test_split

p = yaml.safe_load(open("params.yaml"))["preprocess"]
tr, te = np.load("data/raw/train.npz"), np.load("data/raw/test.npz")
assert tr["x"].shape[1] == p["image_size"]

x = tr["x"].astype("float32") / 255.0
xt = te["x"].astype("float32") / 255.0
mean, std = x.mean(axis=(0, 1, 2)), x.std(axis=(0, 1, 2))
x, xt = (x - mean) / std, (xt - mean) / std
x, xt = x.transpose(0, 3, 1, 2), xt.transpose(0, 3, 1, 2)

xtr, xv, ytr, yv = train_test_split(
    x, tr["y"], test_size=p["val_size"], random_state=p["seed"], stratify=tr["y"])

os.makedirs("data/processed", exist_ok=True)
for n, a, b in [("train", xtr, ytr), ("val", xv, yv), ("test", xt, te["y"])]:
    np.savez(f"data/processed/{n}.npz", x=a.astype("float16"), y=b)  # float16 halves the size