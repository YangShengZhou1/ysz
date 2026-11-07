#!/bin/bash
cd /home/seczone/ysz || exit 1
OUT=outputs/experiments/2026-11-07_next
sleep 420
echo '--- logs ---'
tail -5 "$OUT/allcand_dir.log"
tail -5 "$OUT/allcand_path.log"
echo '--- json files ---'
ls -l "$OUT"/allcand_*.json
echo '--- summaries ---'
for f in "$OUT"/allcand_*_r*.json; do
  echo "== $f"
  cat "$f"
done
