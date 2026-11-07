"""汇总 2026-11-07 两格四项分析结果，输出 summary.json 与紧凑表格。"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

OUT = Path("/home/seczone/ysz/outputs/experiments/2026-11-07_next")
summary = {"kl": {}, "replay": {}}


def rho(k, d):
    ok = np.isfinite(k) & np.isfinite(d) & (np.abs(k) < 1e6)
    if ok.sum() < 3:
        return None
    v = spearmanr(k[ok], d[ok]).statistic
    return float(v) if np.isfinite(v) else None


for p in sorted(OUT.glob("kl_*.json")):
    rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
    k = np.array([r["kl_sym"] for r in rows], float)
    d = np.array([r["delta_acc"] for r in rows], float)
    deg = np.array([r["degenerate_head"] for r in rows], float)
    by = {}
    for r in rows:
        by.setdefault(r["target"], []).append(r)
    ranges, regrets, hits, two_hop, trho = [], [], [], [], []
    for sub in by.values():
        ds = np.array([x["delta_acc"] for x in sub])
        ks = np.array([x["kl_sym"] for x in sub])
        ranges.append((ds.max() - ds.min()) * 100)
        top1 = min(sub, key=lambda x: x["kl_sym"])
        best = max(sub, key=lambda x: x["delta_acc"])
        regrets.append((best["delta_acc"] - top1["delta_acc"]) * 100)
        hits.append(int(best["candidate"] == top1["candidate"]))
        two_hop.append(int(best["hop"] <= 2))
        r_ = rho(ks, ds)
        if r_ is not None:
            trho.append(r_)
    hop_stats = []
    for h in sorted(set(r["hop"] for r in rows)):
        sub = [r for r in rows if r["hop"] == h]
        hop_stats.append([h, len(sub), round(float(np.mean([r["delta_acc"] for r in sub]) * 100), 2)])
    overall = rho(k, d)
    summary["kl"][p.stem] = {
        "n_pairs": len(rows), "n_targets": len(by),
        "degenerate_frac": round(float(deg.mean()), 4),
        "zero_frac": round(float(np.mean(np.abs(d) < 1e-12)), 4),
        "mean_delta_pp": round(float(d.mean() * 100), 2),
        "range_pp_mean": round(float(np.mean(ranges)), 2),
        "regret_pp_mean": round(float(np.mean(regrets)), 2),
        "top1_hit": f"{sum(hits)}/{len(hits)}",
        "overall_rho": round(overall, 3) if overall is not None else None,
        "target_rho_mean": round(float(np.mean(trho)), 3) if trho else None,
        "best_two_hop": f"{sum(two_hop)}/{len(two_hop)}",
        "hop_mean_delta_pp": hop_stats,
    }

for p in sorted(OUT.glob("replay_*.json")):
    with p.with_suffix(".csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    n_t = len({r["target"] for r in rows})
    same = np.mean([int(r["select_best_is_kl_top1"])
                    for r in rows if r["branch"] == "none"])
    br = {}
    for name in ("none", "kl_top1", "select_best"):
        sub = [r for r in rows if r["branch"] == name]
        br[name] = {
            "ver_acc_before": round(float(np.mean([float(r["ver_acc_before"]) for r in sub])), 4),
            "ver_acc_after": round(float(np.mean([float(r["ver_acc_after"]) for r in sub])), 4),
        }
    none_after = {r["target"]: float(r["ver_acc_after"])
                  for r in rows if r["branch"] == "none"}
    gains = {}
    for name in ("kl_top1", "select_best"):
        g = [float(r["ver_acc_after"]) - none_after[r["target"]]
             for r in rows if r["branch"] == name]
        gains[name] = {
            "mean_pp": round(float(np.mean(g) * 100), 2),
            "improved": int(sum(1 for x in g if x > 1e-12)),
            "degraded": int(sum(1 for x in g if x < -1e-12)),
            "flat": int(sum(1 for x in g if abs(x) <= 1e-12)),
        }
    summary["replay"][p.stem] = {
        "n_targets": n_t,
        "select_best_is_kl_top1_frac": round(float(same), 3),
        "branches": br,
        "gain_vs_none_after": gains,
    }

(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("=== 即时评估：候选差异 / 伙伴距离 ===")
print("配置 | 退化 | 零收益 | 平均Δpp | 极差pp | regret pp | 命中 | ρ | 目标内ρ | 最优≤2跳")
for key, s in summary["kl"].items():
    print("{} | {:.1%} | {:.1%} | {:+.2f} | {:.2f} | {:.2f} | {} | {} | {} | {}".format(
        key.replace("kl_paper100-next-", ""), s["degenerate_frac"], s["zero_frac"],
        s["mean_delta_pp"], s["range_pp_mean"], s["regret_pp_mean"], s["top1_hit"],
        s["overall_rho"], s["target_rho_mean"], s["best_two_hop"]))
print()
print("=== 短程重放 / 挑验分离（核查集 acc） ===")
print("配置 | 目标 | 不聚合后 | KL伙伴后 | 选择集最优后 | KL增益pp | 选择增益pp | 选择与KL同")
for key, s in summary["replay"].items():
    b = s["branches"]
    g = s["gain_vs_none_after"]
    print("{} | {} | {:.3f} | {:.3f} | {:.3f} | {:+.2f} | {:+.2f} | {:.1%}".format(
        key.replace("replay_paper100-next-", ""), s["n_targets"],
        b["none"]["ver_acc_after"], b["kl_top1"]["ver_acc_after"],
        b["select_best"]["ver_acc_after"],
        g["kl_top1"]["mean_pp"], g["select_best"]["mean_pp"],
        s["select_best_is_kl_top1_frac"]))
print()
print("=== 按跳数平均收益 (pp) ===")
for key, s in summary["kl"].items():
    print(key.replace("kl_paper100-next-", ""), s["hop_mean_delta_pp"])
