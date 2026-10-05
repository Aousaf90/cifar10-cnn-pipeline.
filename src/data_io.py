import glob
import os
import numpy as np


def save_split(folder, name, x, y, n_shards):
    """Save x, y as n_shards small .npz files: <folder>/<name>_0.npz, ..."""
    os.makedirs(folder, exist_ok=True)
    for i, (a, b) in enumerate(zip(np.array_split(x, n_shards), np.array_split(y, n_shards))):
        np.savez(f"{folder}/{name}_{i}.npz", x=a, y=b)


def load_split(folder, name):
    """Load and concatenate all shards of a split, in order."""
    files = sorted(glob.glob(f"{folder}/{name}_*.npz"), key=lambda f: int(f.rsplit("_", 1)[1][:-4]))
    parts = [np.load(f) for f in files]
    return np.concatenate([p["x"] for p in parts]), np.concatenate([p["y"] for p in parts])