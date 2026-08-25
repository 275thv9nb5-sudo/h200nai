"""
Generate final_report.md from training outputs
================================================
Reads outputs/expA & expB (results.csv + eval txts) and writes a
self-contained final report at the package root, including:
  - per-experiment summary (best epoch, mAP, timings)
  - curve samples every 10 epochs
  - per-class AP from eval txts
  - comparison vs local baselines
  - WHAT TO SEND BACK section (回传清单)

Usage:
  python3 scripts/make_report.py
"""

import os
import time
from pathlib import Path

ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs"

EXPS = [
    ("expA", "exp_yolo26l_4ch_oversample", "YOLO26l + 小图3x过采样 (固定1280, EQLv2)"),
    ("expB", "exp_yolo26x_4ch_oversample", "YOLO26x + 小图3x过采样 (固定1280, EQLv2)"),
]
BASELINE = {"full": 0.4036, "small": 0.6881, "large": 0.3958}
BASELINE_LB = 54.197


def read_results_csv(exp_dir):
    csv = exp_dir / "train" / "results.csv"
    if not csv.exists():
        return None
    rows = []
    with open(csv, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            rows.append(line.strip().split(","))
    cols = {n: i for i, n in enumerate(header)}
    return cols, rows


def main():
    lines = []
    lines.append("# H200 训练结果报告")
    lines.append(f"\n生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n## 实验总览")
    lines.append("\n| 实验 | 模型 | 数据集 | best epoch | mAP50-95 (full) | mAP50-95 (small) | mAP50-95 (large) | 训练时长 |")
    lines.append("|------|------|--------|-----------|-----------------|------------------|------------------|---------|")

    for key, dirname, desc in EXPS:
        exp_dir = OUT / dirname
        got = read_results_csv(exp_dir)
        if got is None:
            lines.append(f"| {key} | {desc.split('+')[0].strip()} | 小图3x过采样 | — | — | — | — | 未运行 |")
            continue
        cols, rows = got
        best_idx = max(range(len(rows)), key=lambda i: float(rows[i][cols["metrics/mAP50-95(B)"]]))
        best_ep = int(float(rows[best_idx][cols["epoch"]]))
        best_map = float(rows[best_idx][cols["metrics/mAP50-95(B)"]])
        final_t = float(rows[-1][cols["time"]])
        # eval txt (small/large)
        ev_txt = OUT / f"eval_{key}.txt"
        m_small = m_large = "—"
        if ev_txt.exists():
            for line in ev_txt.read_text(encoding="utf-8").splitlines():
                if line.startswith("small ("):
                    m_small = line.split("mAP50-95=")[1].split(" ")[0]
                if line.startswith("large ("):
                    m_large = line.split("mAP50-95=")[1].split(" ")[0]
        lines.append(f"| {key} | {desc.split('+')[0].strip()} | 小图3x过采样 | {best_ep}/{len(rows)} | "
                     f"{best_map:.4f} | {m_small} | {m_large} | {final_t/3600:.1f}h |")

    lines.append(f"\n**本地基准 (eqlv2 固定尺度, LB {BASELINE_LB})**: "
                 f"full {BASELINE['full']} / small {BASELINE['small']} / large {BASELINE['large']}")

    # Per-experiment curve
    for key, dirname, desc in EXPS:
        got = read_results_csv(OUT / dirname)
        if got is None:
            continue
        cols, rows = got
        lines.append(f"\n## {key}: {desc}")
        lines.append("\n| epoch | mAP50 | mAP50-95 | box_loss | cls_loss | lr |")
        lines.append("|-------|-------|----------|----------|----------|----|")
        for r in rows:
            ep = int(float(r[cols["epoch"]]))
            if ep % 10 == 0 or ep == len(rows):
                lines.append(f"| {ep} | {float(r[cols['metrics/mAP50(B)']]):.4f} | "
                             f"{float(r[cols['metrics/mAP50-95(B)']]):.4f} | "
                             f"{float(r[cols['train/box_loss']]):.3f} | "
                             f"{float(r[cols['train/cls_loss']]):.3f} | "
                             f"{float(r[cols['lr/pg0']]):.6f} |")

        ev_txt = OUT / f"eval_{key}.txt"
        if ev_txt.exists():
            lines.append(f"\n### {key} 每类 AP (来自 eval_{key}.txt)")
            lines.append("\n```")
            lines.append(ev_txt.read_text(encoding="utf-8"))
            lines.append("```")

    # What to send back
    lines.append("\n## 📦 回传清单（只需要这些）")
    lines.append("\n在包目录执行：`bash pack_results.sh`，把生成的 `results_back.tar.gz` 发回即可。")
    lines.append("\n等价的手动清单：")
    lines.append("\n- `final_report.md`（本文件）")
    lines.append("- `outputs/*/train/results.csv`（训练曲线原始数据）")
    lines.append("- `outputs/*/train/weights/best.pt`（最终模型权重）")
    lines.append("- `outputs/*/train/weights/last.pt`（断点，便于后续续训）")
    lines.append("- `outputs/eval_*.txt`（分组评估结果）")
    lines.append("- `outputs/*/progress_log.txt`（逐轮监测日志）")
    lines.append("\n其余文件（数据集、预训练权重等）无需回传。")

    out_md = ROOT / "final_report.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] written: {out_md}")


if __name__ == '__main__':
    main()
