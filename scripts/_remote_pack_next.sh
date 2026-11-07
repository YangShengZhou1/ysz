#!/bin/bash
# 按 11-05 指导第五节整理交付材料：快照元数据、训练与分析日志
set -u
cd /home/seczone/ysz || exit 1
OUT=outputs/experiments/2026-11-07_next
mkdir -p "$OUT/metadata" "$OUT/logs"

for ck in outputs/checkpoints/*paper100-next*/round_*; do
  case "$ck" in
    *next-dir*) t=dir ;;
    *) t=path ;;
  esac
  r=$(basename "$ck" | sed 's/round_/r/')
  cp "$ck/metadata.json" "$OUT/metadata/${t}_${r}_metadata.json"
done

cp outputs/console/paper100-next-dir.log "$OUT/logs/train_dir.log"
cp outputs/console/paper100-next-path.log "$OUT/logs/train_path.log"
cp "$OUT"/kl_*.log "$OUT/logs/"
cp "$OUT"/replay_*.log "$OUT/logs/"
cp "$OUT/kl_all.log" "$OUT/replay_all.log" "$OUT/logs/"

echo '--- metadata ---'
ls -l "$OUT/metadata/"
echo '--- logs size ---'
du -sh "$OUT/logs"
du -sh "$OUT"
