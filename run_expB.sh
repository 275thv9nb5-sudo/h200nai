#!/bin/bash
# 只跑实验B (YOLO26x + 小图过采样)，约 6-12 小时
set -e
cd "$(dirname "$0")"
export AIC_ROOT="$(pwd)"

bash setup_env.sh
python3 scripts/build_oversample_dataset.py

nohup python3 scripts/monitor_training.py --exp expB --total 100 \
    > outputs/monitor_expB.log 2>&1 &
MON=$!
python3 scripts/train_yolo26x_oversample.py
kill $MON 2>/dev/null || true
python3 scripts/evaluate_best.py --exp expB
python3 scripts/make_report.py
echo "实验B完成，执行 bash pack_results.sh 打包结果"
