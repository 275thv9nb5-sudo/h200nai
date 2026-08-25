#!/bin/bash
# 只跑实验A (YOLO26l + 小图过采样)，约 3-6 小时
set -e
cd "$(dirname "$0")"
export AIC_ROOT="$(pwd)"

bash setup_env.sh
python3 scripts/build_oversample_dataset.py

nohup python3 scripts/monitor_training.py --exp expA --total 100 \
    > outputs/monitor_expA.log 2>&1 &
MON=$!
python3 scripts/train_yolo26l_oversample.py
kill $MON 2>/dev/null || true
python3 scripts/evaluate_best.py --exp expA
python3 scripts/make_report.py
echo "实验A完成，执行 bash pack_results.sh 打包结果"
