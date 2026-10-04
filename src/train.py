import os, yaml, numpy as np, pandas as pd, torch, torch.nn as nn
from models import build_model

p = yaml.safe_load(open("params.yaml"))["train"]
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def load(n):
    d = np.load(f"data/processed/{n}.npz")
    return torch.tensor(d["x"]).float(), torch.tensor(d["y"]).long()

xtr, ytr = load("train"); xv, yv = load("val")
model = build_model(p["num_filters"], p["dropout_rate"]).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
lossf, bs, hist = nn.CrossEntropyLoss(), p["batch_size"], []

for ep in range(p["epochs"]):
    model.train(); perm = torch.randperm(len(xtr)); tot = 0
    for i in range(0, len(xtr), bs):
        idx = perm[i:i + bs]; xb, yb = xtr[idx].to(dev), ytr[idx].to(dev)
        xb = torch.where(torch.rand(len(xb), 1, 1, 1, device=dev) < 0.5, xb.flip(3), xb)  # flip augmentation
        opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
        tot += loss.item() * len(xb)
    model.eval()
    with torch.no_grad():
        out = torch.cat([model(xv[i:i+1000].to(dev)).cpu() for i in range(0, len(xv), 1000)])
    hist.append(dict(epoch=ep + 1, train_loss=tot / len(xtr), val_loss=lossf(out, yv).item(),
                     val_acc=(out.argmax(1) == yv).float().mean().item()))
    print(hist[-1])

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model.pth")
pd.DataFrame(hist).to_csv("models/history.csv", index=False)