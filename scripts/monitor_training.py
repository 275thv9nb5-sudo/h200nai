"""
Monitor training progress — per-epoch log lines + completion detection
========================================================================
Watches outputs/<exp>/train/results.csv and appends one readable line per
new epoch to outputs/<exp>/progress_log.txt. Exits 0 when the target epoch
count is reached, exits 2 if training appears stalled (>30 min no new
rows and no training process).

Usage (run with nohup alongside training):
  python3 scripts/monitor_training.py --exp expA --total 100 [--interval 60]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parent.parent)))

EXP_DIRS = {
    "expA": "exp_yolo26l_4ch_oversample",
    "expB": "exp_yolo26x_4ch_oversample",
}


def find_train_proc():
    try:
        out = subprocess.run(["pgrep", "-f", "train_yolo26"], capture_output=True,
                             text=True).stdout.strip()
        return bool(out)
    except Exception:
        return True  # can't check (e.g. Windows); assume alive


def read_rows(csv_path):
    if not csv_path.exists():
        return "", []
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        header = f.readline().strip()
        for line in f:
            rows.append(line.strip())
    return header, rows


def fmt_time(sec):
    h, m = int(sec // 3600), int(sec % 3600 // 60)
    return f"{h}h{m:02d}m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="expA / expB")
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()

    exp_dir = ROOT / "outputs" / EXP_DIRS[args.exp]
    csv_path = exp_dir / "train" / "results.csv"
    log_path = exp_dir / "progress_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    n_logged = 0
    best_map = 0.0
    best_ep = 0
    t_start = time.time()
    last_growth = time.time()

    print(f"[monitor:{args.exp}] watching {csv_path} (target {args.total} epochs)")
    if not csv_path.exists():
        print(f"[monitor:{args.exp}] results.csv not created yet, waiting...")

    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"=== {args.exp} monitor started {time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"(target {args.total} epochs) ===\n")

        while True:
            header, rows = read_rows(csv_path)
            cols = {name: i for i, name in enumerate(header.split(","))}
            for r in rows[n_logged:]:
                c = r.split(",")
                ep = int(float(c[0]))
                m50 = float(c[cols["metrics/mAP50(B)"]])
                m5095 = float(c[cols["metrics/mAP50-95(B)"]])
                box = float(c[cols["train/box_loss"]])
                cls = float(c[cols["train/cls_loss"]])
                lr = float(c[cols["lr/pg0"]])
                t = float(c[cols["time"]])
                if m5095 > best_map:
                    best_map, best_ep = m5095, ep
                eta = fmt_time(t / max(ep, 1) * (args.total - ep))
                line = (f"[{time.strftime('%m-%d %H:%M:%S')}] ep {ep:3d}/{args.total} | "
                        f"mAP50={m50:.4f} mAP50-95={m5095:.4f} (best ep{best_ep}={best_map:.4f}) | "
                        f"box={box:.3f} cls={cls:.3f} | lr={lr:.6f} | "
                        f"elapsed={fmt_time(t)} | ETA~{eta}")
                log.write(line + "\n")
                log.flush()
                n_logged += 1
                last_growth = time.time()

            if len(rows) >= args.total:
                line = (f"[{time.strftime('%m-%d %H:%M:%S')}] DONE: {len(rows)}/{args.total} "
                        f"epochs, best ep{best_ep} mAP50-95={best_map:.4f}")
                log.write(line + "\n")
                print(f"[monitor:{args.exp}] {line}")
                return 0

            if time.time() - last_growth > 1800 and not find_train_proc():
                line = (f"[{time.strftime('%m-%d %H:%M:%S')}] STALLED: no new epochs for "
                        f">30min and training process gone")
                log.write(line + "\n")
                print(f"[monitor:{args.exp}] {line}")
                return 2

            time.sleep(args.interval)


if __name__ == '__main__':
    sys.exit(main())
