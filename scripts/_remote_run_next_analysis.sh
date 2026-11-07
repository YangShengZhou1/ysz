#!/bin/bash
# 两格论文级快照的四项离线分析：即时评估（GPU2）与重放/挑验分离（GPU3）并行
set -u
cd /home/seczone/ysz || exit 1
export LD_LIBRARY_PATH=/home/seczone/anaconda3/lib
PY=/home/seczone/anaconda3/bin/python
DATA=/home/seczone/ysz/datasets/cifar10
OUT=outputs/experiments/2026-11-07_next
mkdir -p "$OUT"

TAGS="paper100-next-dir paper100-next-path"

kl_all() {
  for tag in $TAGS; do
    base=$(ls -d outputs/checkpoints/*${tag}* | head -1)
    for r in 1 250 500; do
      echo "[kl] $tag r$r $(date +%T)"
      "$PY" scripts/kl_benefit_distribution.py \
        --checkpoint "$base/round_$r" --data_dir "$DATA" \
        --output "$OUT/kl_${tag}_r${r}.json" \
        --device cuda:2 --scope base --max_candidates 20 --dataset cifar10 \
        > "$OUT/kl_${tag}_r${r}.log" 2>&1
      echo "[kl] $tag r$r rc=$?"
    done
  done
}

replay_all() {
  for tag in $TAGS; do
    base=$(ls -d outputs/checkpoints/*${tag}* | head -1)
    for r in 250 500; do
      echo "[replay] $tag r$r $(date +%T)"
      timeout 7200 "$PY" scripts/same_target_replay_paper100.py \
        --checkpoint "$base/round_$r" --data_dir "$DATA" \
        --output "$OUT/replay_${tag}_r${r}" \
        --device cuda:3 --dataset cifar10 --n_candidates 4 --replay_epochs 1 \
        --batch_size 128 --eval_batch_size 256 \
        > "$OUT/replay_${tag}_r${r}.log" 2>&1
      echo "[replay] $tag r$r rc=$?"
    done
  done
}

kl_all > "$OUT/kl_all.log" 2>&1 &
P1=$!
replay_all > "$OUT/replay_all.log" 2>&1 &
P2=$!
wait "$P1" "$P2"
echo "ALL_DONE $(date +%T)" >> "$OUT/kl_all.log"
ls -l "$OUT"/*.json "$OUT"/*.csv 2>/dev/null
