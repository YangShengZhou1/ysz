#!/bin/bash
# 补齐全部候选的 1-epoch 重放输出，两格并行分别用 GPU2 / GPU3
set -u
cd /home/seczone/ysz || exit 1
export LD_LIBRARY_PATH=/home/seczone/anaconda3/lib
PY=/home/seczone/anaconda3/bin/python
DATA=/home/seczone/ysz/datasets/cifar10
OUT=outputs/experiments/2026-11-07_next
mkdir -p "$OUT"

run_one() {
  tag="$1"; gpu="$2"
  for r in 250 500; do
    ckpt=$(ls -d outputs/checkpoints/*${tag}*/round_${r} | head -1)
    echo "[allcand] $tag r$r $(date +%T)"
    timeout 7200 "$PY" scripts/same_target_replay_all_candidates.py \
      --checkpoint "$ckpt" --data_dir "$DATA" \
      --output "$OUT/allcand_${tag}_r${r}" \
      --device "cuda:${gpu}" --dataset cifar10 \
      --n_candidates 4 --replay_epochs 1 \
      --batch_size 128 --eval_batch_size 256 \
      > "$OUT/allcand_${tag}_r${r}.log" 2>&1
    echo "[allcand] $tag r$r rc=$?"
  done
}

run_one paper100-next-dir 2 > "$OUT/allcand_dir.log" 2>&1 &
P1=$!
run_one paper100-next-path 3 > "$OUT/allcand_path.log" 2>&1 &
P2=$!
wait "$P1" "$P2"
echo "ALLCAND_DONE $(date +%T)" >> "$OUT/allcand_dir.log"
