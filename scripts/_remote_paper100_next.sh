#!/bin/bash
# 第二轮：按 11-05 指导方案，为「同目标候选比较 + 短程重放 + 选择/核查分离」准备论文级快照。
# 两格均为论文 Experiment Setup 中的配置，除划分方式外其余十项与第一趟 (RUN-024) 完全一致。
set -u
cd /home/seczone/ysz || exit 1
export LD_LIBRARY_PATH=/home/seczone/anaconda3/lib
PY=/home/seczone/anaconda3/bin/python
ENTRY=fedml_experiments/standalone/ProToDFL/main_ProToDFL.py
mkdir -p outputs/console

run_one() {
  TAG="$1"; PART="$2"; ALPHA="$3"; GPU="$4"
  CUDA_VISIBLE_DEVICES="$GPU" setsid nohup timeout 129600 "$PY" "$ENTRY" \
    --dataset cifar10 --model resnet18 \
    --data_dir datasets --output_dir outputs \
    --partition_method "$PART" --partition_alpha "$ALPHA" \
    --client_num_in_total 100 --frac 1.0 \
    --comm_round 500 --epochs 5 --batch_size 128 \
    --lr 0.1 --lr_decay 1.0 \
    --cs random --dynamic 0 --k 2 --t_max 100 \
    --times 1 --step 0 \
    --max_neighbors 0 --validation_ratio 0.2 \
    --save_models_round 1,250,500 \
    --gpu 0 --seed 0 --aggregate_scope base --tag "$TAG" \
    > "outputs/console/${TAG}.log" 2>&1 &
  echo "started $TAG on GPU $GPU"
}

run_one paper100-next-dir my_part 0.3 0
run_one paper100-next-path n_cls 2 1
sleep 3
