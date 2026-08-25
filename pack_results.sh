#!/bin/bash
# 打包需要回传的结果文件 -> results_back.tar.gz
# 在包目录下执行: bash pack_results.sh
# 把生成的 results_back.tar.gz 发给机主即可（其余文件不需要）
set -e
cd "$(dirname "$0")"

python3 scripts/make_report.py   # 先刷新 final_report.md

FILES=(
    final_report.md
)
# 动态收集存在的输出文件
for exp in outputs/exp_*/train; do
    [ -d "$exp" ] || continue
    [ -f "$exp/results.csv" ] && FILES+=("$exp/results.csv")
    [ -f "$exp/weights/best.pt" ] && FILES+=("$exp/weights/best.pt")
    [ -f "$exp/weights/last.pt" ] && FILES+=("$exp/weights/last.pt")
done
for f in outputs/eval_*.txt outputs/*/progress_log.txt; do
    [ -f "$f" ] && FILES+=("$f")
done

tar -czf results_back.tar.gz "${FILES[@]}"
echo "=============================================="
echo "打包完成: results_back.tar.gz ($(du -h results_back.tar.gz | cut -f1))"
echo "把这个文件发回即可，其余文件无需回传。"
echo "=============================================="
