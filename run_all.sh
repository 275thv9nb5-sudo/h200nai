#!/bin/bash
# 一键运行：环境检查 -> 构建过采样数据集 -> 实验A -> 实验B -> 评估 -> 最终报告
# 预计总时长 (H200): 实验A 3-6h + 实验B 6-12h
#
# 用法:
#   bash run_all.sh                # 前台运行（SSH 断开会中断，不推荐）
#   nohup bash run_all.sh > outputs/run_all.log 2>&1 &   # 推荐：后台运行
#
# 容错设计: 任一实验失败不会阻塞下一个实验 (fail-fast 仅用于环境/数据阶段),
# 失败信息记录在 run_all.log, final_report.md 会标明未运行的实验。
set -e
cd "$(dirname "$0")"
export AIC_ROOT="$(pwd)"

echo "=============================================="
echo "AIC H200 训练包 - 开始 $(date '+%F %T')"
echo "=============================================="

# 环境/数据阶段仍用 fail-fast: 这两步坏了后面全白跑
bash setup_env.sh
python3 scripts/build_oversample_dataset.py

run_experiment() {
    local key="$1" script="$2"
    echo ""
    echo "===== 实验${key}: 开始 $(date '+%F %T') ====="
    nohup python3 scripts/monitor_training.py --exp "${key}" --total 100 \
        > "outputs/monitor_${key}.log" 2>&1 &
    local mon=$!
    local rc=0
    python3 "scripts/${script}" || rc=$?
    kill ${mon} 2>/dev/null || true
    if [ ${rc} -eq 0 ]; then
        python3 scripts/evaluate_best.py --exp "${key}" \
            || echo "[实验${key}] 评估失败 (exit $?) - 训练本身已完成"
        echo "===== 实验${key}: 完成 $(date '+%F %T') ====="
    else
        echo "!!!!! 实验${key}: 训练失败 (exit ${rc})，继续下一个实验。"
        echo "!!!!! 详见 outputs/monitor_${key}.log 与 outputs/run_all.log"
    fi
}

# ---------- 实验 A: YOLO26l + 小图过采样 ----------
run_experiment expA train_yolo26l_oversample.py

# ---------- 实验 B: YOLO26x + 小图过采样 ----------
run_experiment expB train_yolo26x_oversample.py

# ---------- 最终报告 ----------
python3 scripts/make_report.py || echo "报告生成失败，可稍后手动执行 python3 scripts/make_report.py"

echo "=============================================="
echo "全部流程结束 $(date '+%F %T')"
echo "请执行: bash pack_results.sh 打包结果发回"
echo "=============================================="
