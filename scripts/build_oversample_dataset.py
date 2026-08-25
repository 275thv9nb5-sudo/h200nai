"""
Build oversampled 4ch dataset (small images 3x) — pure file copies
========================================================================
Small images (360x640, identified by PNG header width<=640) make up only
96/1657 = 5.8% of the rebalanced train split, but 15.5% of the test set.
This script copies each small image + its label 2 extra times
(96 -> 288 = 15.6% of 1849 train images, aligned with the test mix).

Val split is copied unchanged.

Idempotent: skips if dataset_oversample/images/train already exists.
Runtime: ~1-2 minutes. No dependencies beyond Python stdlib.

Usage:
  python3 scripts/build_oversample_dataset.py
"""

import os
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parent.parent)))
SRC = ROOT / "dataset"
DST = ROOT / "dataset_oversample"
OV_FACTOR = 3  # 3x total copies of each small image (1 original + 2 extra)


def png_size(path):
    """Read PNG (width, height) from IHDR without decoding."""
    with open(path, "rb") as f:
        head = f.read(33)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}"
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def main():
    assert (SRC / "images/train").exists(), f"missing {SRC}/images/train"

    if (DST / "images/train").exists() and any((DST / "images/train").iterdir()):
        n = len(os.listdir(DST / "images/train"))
        print(f"[Skip] {DST}/images/train already exists ({n} files)")
        return

    for split in ("train", "val"):
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1) Copy val unchanged
    n_val = 0
    for f in os.listdir(SRC / "images/val"):
        if not f.endswith(".png"):
            continue
        stem = f[:-4]
        shutil.copy2(str(SRC / "images/val" / f), str(DST / "images/val" / f))
        shutil.copy2(str(SRC / "labels/val" / f"{stem}.txt"),
                     str(DST / "labels/val" / f"{stem}.txt"))
        n_val += 1
    print(f"[OK] val copied: {n_val} images (unchanged)")

    # 2) Copy train; oversample small images
    n_large = n_small = n_extra = 0
    for f in sorted(os.listdir(SRC / "images/train")):
        if not f.endswith(".png"):
            continue
        stem = f[:-4]
        w, h = png_size(SRC / "images/train" / f)
        small = w <= 640

        shutil.copy2(str(SRC / "images/train" / f), str(DST / "images/train" / f))
        shutil.copy2(str(SRC / "labels/train" / f"{stem}.txt"),
                     str(DST / "labels/train" / f"{stem}.txt"))

        if small:
            n_small += 1
            for k in range(1, OV_FACTOR):
                new_stem = f"{stem}__ov{k}"
                shutil.copy2(str(SRC / "images/train" / f),
                             str(DST / "images/train" / f"{new_stem}.png"))
                shutil.copy2(str(SRC / "labels/train" / f"{stem}.txt"),
                             str(DST / "labels/train" / f"{new_stem}.txt"))
                n_extra += 1
        else:
            n_large += 1

    n_train = len(os.listdir(DST / "images/train"))
    print(f"[OK] train: {n_large} large + {n_small} small (x{OV_FACTOR} = {n_small*OV_FACTOR}) "
          f"+ {n_extra} extra copies = {n_train} total")
    print(f"     small share = {n_small*OV_FACTOR}/{n_train} = "
          f"{n_small*OV_FACTOR/n_train*100:.1f}%  (test = 15.5%)")

    # Label integrity check
    for split in ("train", "val"):
        imgs = {f[:-4] for f in os.listdir(DST / "images" / split) if f.endswith(".png")}
        lbls = {f[:-4] for f in os.listdir(DST / "labels" / split) if f.endswith(".txt")}
        assert imgs == lbls, f"label mismatch in {split}"
    print("[OK] dataset_oversample ready")


if __name__ == '__main__':
    main()
