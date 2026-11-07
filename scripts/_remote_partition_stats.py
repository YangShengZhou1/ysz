"""老师 11-05 第四节第 1 条：统计两格划分下每客户端样本量与各类别样本数。"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/seczone/ysz")
DATA = ROOT / "datasets/cifar10"
OUT = ROOT / "outputs/experiments/2026-11-07_next"
OUT.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT))
from fedml_api.data_preprocessing.cifar10.data_loader import CIFAR10_truncated

full = CIFAR10_truncated(str(DATA), train=True, download=False)
labels = np.asarray(full.target)

result = {}
for tag in ("paper100-next-dir", "paper100-next-path"):
    ck = sorted((ROOT / "outputs/checkpoints").glob(f"*{tag}*/round_500"))[0]
    meta = json.loads((ck / "metadata.json").read_text(encoding="utf-8"))
    per_client = []
    for c in meta["clients"]:
        idx = np.asarray(c["train_indices"], dtype=int)
        counts = np.bincount(labels[idx], minlength=10)
        per_client.append({
            "client_idx": c["client_idx"],
            "train_n": len(idx),
            "class_counts": counts.tolist(),
            "nonzero_classes": int((counts > 0).sum()),
        })
    sizes = np.array([x["train_n"] for x in per_client])
    nonzero = np.array([x["nonzero_classes"] for x in per_client])
    result[tag] = {
        "checkpoint": str(ck),
        "partition_method": meta["args"].get("partition_method"),
        "partition_alpha": meta["args"].get("partition_alpha"),
        "n_clients": len(per_client),
        "train_n_min": int(sizes.min()),
        "train_n_max": int(sizes.max()),
        "train_n_mean": float(sizes.mean()),
        "nonzero_classes_min": int(nonzero.min()),
        "nonzero_classes_max": int(nonzero.max()),
        "nonzero_classes_mean": float(nonzero.mean()),
        "per_client": per_client,
    }
    print(tag, "| 样本量", sizes.min(), "-", sizes.max(), "均值", round(sizes.mean(), 1),
          "| 每客户端非零类别数", nonzero.min(), "-", nonzero.max(), "均值", round(nonzero.mean(), 2))

(OUT / "partition_stats.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("written:", OUT / "partition_stats.json")
