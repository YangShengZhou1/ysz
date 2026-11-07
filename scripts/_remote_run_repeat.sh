#!/bin/bash
# 固定 10 个客户端(每格 0-4), 三分支各重复 3 遍, 两格并行用 GPU2 / GPU3
set -u
cd /home/seczone/ysz || exit 1
export LD_LIBRARY_PATH=/home/seczone/anaconda3/lib
PY=/home/seczone/anaconda3/bin/python
DATA=/home/seczone/ysz/datasets/cifar10
OUT=outputs/experiments/2026-11-07_repeat
mkdir -p "$OUT"

run_one() {
  tag="$1"; gpu="$2"
  for r in 250 500; do
    ckpt=$(ls -d outputs/checkpoints/*${tag}*/round_${r} | head -1)
    echo "[repeat] $tag r$r $(date +%T)"
    timeout 3600 "$PY" scripts/repeat_replay_check.py \
      --checkpoint "$ckpt" --data_dir "$DATA" \
      --output "$OUT/repeat_${tag}_r${r}" \
      --device "cuda:${gpu}" --targets 0,1,2,3,4 --repeats 3 \
      --batch_size 128 --eval_batch_size 256 \
      > "$OUT/repeat_${tag}_r${r}.log" 2>&1
    echo "[repeat] $tag r$r rc=$?"
  done
}

run_one paper100-next-dir 2 > "$OUT/dir.log" 2>&1 &
P1=$!
run_one paper100-next-path 3 > "$OUT/path.log" 2>&1 &
P2=$!
wait "$P1" "$P2"
echo "REPEAT_DONE $(date +%T)" >> "$OUT/dir.log"
