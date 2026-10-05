import os
import yaml
import torch
import torch.nn as nn
import pandas as pd
from data_io import load_split
from models import build_model

p = yaml.safe_load(open("params.yaml"))["train"]
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load(name):
    x, y = load_split("data/processed", name)
    return torch.tensor(x).float(), torch.tensor(y).long()


xtr, ytr = load("train")
xv, yv = load("val")

model = build_model(p["num_filters"], p["dropout_rate"]).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
lossf, bs, hist = nn.CrossEntropyLoss(), p["batch_size"], []

for ep in range(p["epochs"]):
    model.train()
    perm, total = torch.randperm(len(xtr)), 0.0
    for i in range(0, len(xtr), bs):
        idx = perm[i:i + bs]
        xb, yb = xtr[idx].to(dev), ytr[idx].to(dev)
        flip = torch.rand(len(xb), device=dev) < 0.5                 # random horizontal flip
        xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
        opt.zero_grad()
        loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)

    model.eval()
    with torch.no_grad():
        out = torch.cat([model(xv[i:i + 1000].to(dev)).cpu() for i in range(0, len(xv), 1000)])
    row = dict(epoch=ep + 1, train_loss=total / len(xtr),
               val_loss=lossf(out, yv).item(), val_acc=(out.argmax(1) == yv).float().mean().item())
    hist.append(row)
    print(row, flush=True)

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model.pth")
pd.DataFrame(hist).to_csv("models/history.csv", index=False)