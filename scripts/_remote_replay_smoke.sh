#!/bin/bash
# 用新落盘的 r1 快照跑 2 个目标的小样，验证重放脚本可用性（GPU2，不干扰训练）
cd /home/seczone/ysz || exit 1
export LD_LIBRARY_PATH=/home/seczone/anaconda3/lib
PY=/home/seczone/anaconda3/bin/python
OUT=outputs/experiments/2026-11-06_replay
mkdir -p "$OUT"
CKPT=$(ls -d outputs/checkpoints/*paper100-next-dir*/round_1 | head -1)
echo "checkpoint: $CKPT"
date +%T
timeout 1800 "$PY" scripts/same_target_replay_paper100.py \
  --checkpoint "$CKPT" \
  --data_dir /home/seczone/ysz/datasets/cifar10 \
  --output "$OUT/smoke_dir_r1" \
  --device cuda:2 --dataset cifar10 \
  --max_targets 2 --n_candidates 4 --replay_epochs 1 \
  --batch_size 128 --eval_batch_size 256
echo "rc=$?"
date +%T
ls -l "$OUT"/smoke_dir_r1.* 2>/dev/null
