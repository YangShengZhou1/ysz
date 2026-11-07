"""老师 11-05 第 45 行：核对各分支起点只在计划改变的聚合部分存在差异。"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/seczone/ysz")
sys.path.insert(0, str(ROOT / "scripts"))
from paired_aggregation_check import aggregate

ck = sorted((ROOT / "outputs/checkpoints").glob("*paper100-next-dir*/round_500"))[0]
meta = json.loads((ck / "metadata.json").read_text(encoding="utf-8"))
clients = {int(c["client_idx"]): c for c in meta["clients"]}
states = {i: torch.load(c["model_file"], map_location="cpu", weights_only=True)
          for i, c in clients.items()}
ns = {i: int(c["train_num"]) for i, c in clients.items()}


def same(a, b):
    return torch.equal(a, b)


report = []
for target in sorted(clients)[:5]:
    neighbor = sorted(c["selected_neighbors"] for c in [clients[target]])[0][0]
    base = states[target]
    none = {k: v.detach().clone() for k, v in base.items()}
    agg = aggregate(base, [states[neighbor]], ns[target], [ns[neighbor]], "base")

    head = [k for k in base if k.startswith("head.") or k.startswith("linear.")]
    bn = [k for k in base if any(x in k for x in ("running_mean", "running_var", "num_batches_tracked"))]
    learn = [k for k in base if k not in head and k not in bn]

    check = {
        "target": target,
        "neighbor": neighbor,
        "n_head_keys": len(head),
        "n_bn_stat_keys": len(bn),
        "n_learn_keys": len(learn),
        "head_same_none_vs_base": all(same(none[k], base[k]) for k in head),
        "head_same_agg_vs_base": all(same(agg[k], base[k]) for k in head),
        "bn_same_none_vs_base": all(same(none[k], base[k]) for k in bn),
        "bn_same_agg_vs_base": all(same(agg[k], base[k]) for k in bn),
        "learn_same_agg_vs_base": all(same(agg[k], base[k]) for k in learn),
        "none_vs_agg_differing_keys": sum(1 for k in base if not same(none[k], agg[k])),
    }
    report.append(check)

print(json.dumps({
    "checkpoint": str(ck),
    "n_checked_targets": len(report),
    "per_target": report,
    "summary": {
        "head_preserved_by_aggregate": all(r["head_same_agg_vs_base"] for r in report),
        "bn_stats_preserved_by_aggregate": all(r["bn_same_agg_vs_base"] for r in report),
        "learned_base_changed_by_aggregate": all(not r["learn_same_agg_vs_base"] for r in report),
    },
}, ensure_ascii=False, indent=2))
