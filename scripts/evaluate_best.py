"""
Evaluate an experiment's best.pt on full / small / large val subsets
========================================================================
Runs ultralytics val() three times (GPU, few minutes on H200) and writes
outputs/eval_<exp>.txt with:
  - full val mAP50 / mAP50-95 + per-class AP50-95
  - small-image subset (360x640) mAP
  - large-image subset (1080x1920) mAP
  - local baselines for comparison (full 0.4036 / small 0.6881 / large 0.3958)

Usage:
  python3 scripts/evaluate_best.py --exp expA
"""

import argparse
import os
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parent.parent)))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_4ch_patch import patch_4ch_loading
patch_4ch_loading()

from ultralytics import YOLO

BASELINES = {"full": 0.4036, "small": 0.6881, "large": 0.3958}
BASELINE_LB = 54.197


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(33)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def build_subset(name, is_small):
    """Copy val images+labels of one size class into outputs/eval_<exp>_<name>."""
    out = ROOT / "outputs" / f"eval_{args.exp}_{name}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    src_img = ROOT / "dataset" / "images" / "val"
    src_lbl = ROOT / "dataset" / "labels" / "val"
    n = 0
    for f in sorted(src_img.glob("*.png")):
        w, h = png_size(f)
        if (w <= 640) != is_small:
            continue
        stem = f.stem
        shutil.copy2(str(f), str(out / "images" / f.name))
        shutil.copy2(str(src_lbl / f"{stem}.txt"), str(out / "labels" / f"{stem}.txt"))
        n += 1
    return out, n


def write_yaml(out):
    import yaml
    yaml_path = out / "data.yaml"
    data = {
        "path": str(out),
        "train": str(out / "images"),
        "val": str(out / "images"),
        "names": {i: f"class_{i}" for i in range(12)},
        "nc": 12,
        "channels": 4,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)
    return yaml_path


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()

    exp_dir = ROOT / "outputs" / args.exp
    best_pt = exp_dir / "train" / "weights" / "best.pt"
    assert best_pt.exists(), f"missing {best_pt}"

    model = YOLO(str(best_pt))
    print(f"[eval:{args.exp}] model loaded: {best_pt}")

    lines = []
    lines.append(f"=== eval {args.exp} ({best_pt}) ===")
    lines.append(f"baselines (eqlv2 fixed-scale, LB {BASELINE_LB}): "
                 f"full {BASELINES['full']} / small {BASELINES['small']} / large {BASELINES['large']}")

    # Full val (use the experiment's own data.yaml -> rebalanced val)
    r = model.val(data=str(exp_dir / "data.yaml"), imgsz=1280, batch=4,
                  device="cuda", verbose=False)
    lines.append(f"full : mAP50={r.box.map50:.4f} mAP50-95={r.box.map:.4f} "
                 f"(baseline {BASELINES['full']})")
    for i in range(len(r.box.maps)):
        lines.append(f"  class_{i:2d}: AP50-95={r.box.maps[i]:.4f}")

    # Small / large subsets
    for name, is_small in (("small", True), ("large", False)):
        out, n = build_subset(name, is_small)
        r = model.val(data=str(write_yaml(out)), imgsz=1280, batch=4,
                      device="cuda", verbose=False)
        lines.append(f"{name} ({n} imgs): mAP50={r.box.map50:.4f} "
                     f"mAP50-95={r.box.map:.4f} (baseline {BASELINES[name]})")

    out_txt = ROOT / "outputs" / f"eval_{args.exp}.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[eval:{args.exp}] written: {out_txt}")
    print("\n".join(lines))


if __name__ == '__main__':
    main()
