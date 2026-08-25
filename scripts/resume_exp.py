"""
Resume an experiment from last.pt (after reboot / interruption)
=================================================================
Deterministic explicit-path resume (NOT resume=True globbing).

Usage:
  python3 scripts/resume_exp.py --exp expA [--batch 8 --workers 8]
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parent.parent)))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eql_v2_loss import install_eql_v2_hook
from dataset_4ch_patch import patch_4ch_loading

CLASS_COUNTS = [5499, 135, 3061, 621, 809, 666, 1571, 89, 1455, 282, 204, 25]
install_eql_v2_hook(class_counts=CLASS_COUNTS, alpha=0.5, gamma=4.0, mu=0.5)
patch_4ch_loading()

from ultralytics import YOLO

EXP_DIRS = {
    "expA": "exp_yolo26l_4ch_oversample",
    "expB": "exp_yolo26x_4ch_oversample",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EXP_DIRS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    exp_dir = ROOT / "outputs" / EXP_DIRS[args.exp]
    last_pt = exp_dir / "train" / "weights" / "last.pt"
    if not last_pt.exists():
        print(f"[ERROR] Checkpoint not found: {last_pt}")
        print("Nothing to resume — start fresh with the train script instead.")
        sys.exit(1)

    # Integrity pre-check (a hard kill during torch.save can truncate)
    import torch
    try:
        torch.load(str(last_pt), map_location="cpu")
    except Exception as e:
        print(f"[ERROR] last.pt failed to load ({type(e).__name__}) - corrupt. "
              f"Re-extract from the original package or re-run training.")
        sys.exit(1)

    print("=" * 60)
    print(f"Resume {args.exp}: {last_pt}")
    print("=" * 60)
    model = YOLO(str(last_pt))
    print("[OK] Checkpoint loaded (model + optimizer + epoch)")

    model.train(
        resume=str(last_pt),
        batch=args.batch,
        workers=args.workers,
        device='cuda',
    )
    print(f"\n[OK] {args.exp} resumed and completed!")
    print(f"Best model: {exp_dir}/train/weights/best.pt")


if __name__ == '__main__':
    main()
