#!/bin/bash
# 环境检查/安装脚本
# - ultralytics 必须 8.4.116（训练脚本依赖其内部接口，勿升级）
# - H200 (Hopper) 需要 torch >= 2.4（cu124/cu126 均可）
set -e
cd "$(dirname "$0")"

echo "[setup] 检查 Python 环境..."
python3 -c "import ultralytics" 2>/dev/null || {
    echo "[setup] 未安装 ultralytics，安装 8.4.116..."
    python3 -m pip install ultralytics==8.4.116
}

python3 - <<'EOF'
import torch, ultralytics, sys
assert torch.cuda.is_available(), "CUDA 不可用，请检查驱动/容器"
v = tuple(int(x) for x in torch.__version__.split(".")[:2])
assert v >= (2, 4), f"torch 过旧: {torch.__version__}（H200 需要 >=2.4）"
assert ultralytics.__version__ == "8.4.116", f"ultralytics 版本错误: {ultralytics.__version__}（必须 8.4.116）"
print(f"[setup] 环境 OK: torch {torch.__version__} | ultralytics {ultralytics.__version__} | {torch.cuda.get_device_name(0)}")
EOF
