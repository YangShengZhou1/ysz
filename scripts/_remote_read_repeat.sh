#!/bin/bash
cd /home/seczone/ysz || exit 1
OUT=outputs/experiments/2026-11-07_repeat
for f in "$OUT"/repeat_*.json; do
  echo "== $f"
  cat "$f"
done
