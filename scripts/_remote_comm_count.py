"""按老师 11-05 第四节要求，统计实际交换张量的字节数口径所需的基本量。"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/seczone/ysz")
CKPT = sorted((ROOT / "outputs/checkpoints").glob("*paper100-next-dir*/round_1"))[0]
meta = json.loads((CKPT / "metadata.json").read_text(encoding="utf-8"))
args = meta["args"]

state = torch.load(meta["clients"][0]["model_file"], map_location="cpu", weights_only=True)
proto = torch.load(meta["clients"][0]["proto_file"], map_location="cpu", weights_only=True)

head_keys = [k for k in state if k.startswith("head.") or k.startswith("linear.")]
base_keys = [k for k in state if k not in head_keys]
base_elems = sum(int(state[k].numel()) for k in base_keys)
head_elems = sum(int(state[k].numel()) for k in head_keys)
proto_elems = int(proto.numel())

n_clients = len(meta["clients"])
edges = len(meta["graph_edges"])
selected = sum(len(c["selected_neighbors"]) for c in meta["clients"])
rounds = int(args.get("comm_round", 500))
bytes_per_elem = 4  # fp32

out = {
    "checkpoint": str(CKPT),
    "partition_method": args.get("partition_method"),
    "client_num_in_total": n_clients,
    "comm_round": rounds,
    "graph_edges": edges,
    "selected_neighbor_slots_total": selected,
    "base_elements": base_elems,
    "base_bytes": base_elems * bytes_per_elem,
    "head_elements": head_elems,
    "head_bytes": head_elems * bytes_per_elem,
    "proto_elements": proto_elems,
    "proto_bytes": proto_elems * bytes_per_elem,
    "bytes_per_exchange_base_only": base_elems * bytes_per_elem,
    "bytes_per_round_if_all_selected_send_base": selected * base_elems * bytes_per_elem,
    "bytes_full_training_if_all_selected_send_base": selected * base_elems * bytes_per_elem * rounds,
    "note": "控制消息、多跳转发、原型之外的元数据未计入；本表只统计稠密张量有效负载。",
}
print(json.dumps(out, ensure_ascii=False, indent=2))
