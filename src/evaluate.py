import json
import yaml
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data_io import load_split
from models import build_model

p = yaml.safe_load(open("params.yaml"))["train"]
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

x, y = load_split("data/processed", "test")
x, y = torch.tensor(x).float(), torch.tensor(y).long()

model = build_model(p["num_filters"], p["dropout_rate"]).to(dev)
model.load_state_dict(torch.load("models/model.pth", map_location=dev))
model.eval()
with torch.no_grad():
    out = torch.cat([model(x[i:i + 1000].to(dev)).cpu() for i in range(0, len(x), 1000)])

pred = out.argmax(1)
metrics = {"test_loss": F.cross_entropy(out, y).item(),
           "test_accuracy": (pred == y).float().mean().item()}
json.dump(metrics, open("metrics.json", "w"), indent=2)
print(metrics)

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
fig, ax = plt.subplots(figsize=(8, 8))
ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=classes).plot(ax=ax, xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png")