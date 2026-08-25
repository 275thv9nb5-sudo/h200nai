"""
Experiment B: YOLO26x 4ch + EQL v2 + small-image oversampling (FIXED 1280)
============================================================================
Capacity scaling: s(10M)=LB52.5, l(26M)=LB54.2 -> x(59M) expected 55.5~56.
Same proven eqlv2 recipe + oversampled small images, NO multi_scale.

Prerequisite: scripts/build_oversample_dataset.py must have run.

Usage:
  python3 scripts/train_yolo26x_oversample.py                  # full (H200)
  python3 scripts/train_yolo26x_oversample.py --batch 2 --workers 0 --smoke
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parent.parent)))

# ============================================================
# INSTALL EQL v2 BEFORE importing ultralytics (CRITICAL)
# ============================================================
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eql_v2_loss import install_eql_v2_hook

CLASS_COUNTS = [5499, 135, 3061, 621, 809, 666, 1571, 89, 1455, 282, 204, 25]
install_eql_v2_hook(
    class_counts=CLASS_COUNTS,
    alpha=0.5,
    gamma=4.0,
    mu=0.5,
)

from ultralytics import YOLO
from dataset_4ch_patch import patch_4ch_loading
patch_4ch_loading()

# ============================================================
# Paths
# ============================================================
DATASET_DIR = ROOT / "dataset_oversample"
OUTPUT_DIR = ROOT / "outputs" / "exp_yolo26x_4ch_oversample"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

n_train = len(os.listdir(DATASET_DIR / "images/train"))
n_val = len(os.listdir(DATASET_DIR / "images/val"))
print(f"\n[Config] Dataset: {n_train} train + {n_val} val (oversampled small images)")
print(f"[Config] Output: {OUTPUT_DIR}")

data_yaml = {
    'path': str(DATASET_DIR),
    'train': str(DATASET_DIR / 'images/train'),
    'val': str(DATASET_DIR / 'images/val'),
    'names': {i: f'class_{i}' for i in range(12)},
    'nc': 12,
    'channels': 4,  # CRITICAL: without this ultralytics silently rebuilds as 3ch!
}
yaml_path = OUTPUT_DIR / "data.yaml"
with open(yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(data_yaml, f, allow_unicode=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    model = YOLO(str(ROOT / 'yolo26x.pt'))
    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"[Model] YOLO26x: {n_params:.1f}M params")

    # First conv: 3ch->4ch (Blue->IR weight copy + 1% noise)
    old_conv = model.model.model[0].conv
    old_weight = old_conv.weight.data
    out_ch = old_weight.shape[0]

    new_conv = nn.Conv2d(4, out_ch, 3, stride=2, padding=1,
                         bias=(old_conv.bias is not None))
    with torch.no_grad():
        new_conv.weight[:, :3] = old_weight
        new_conv.weight[:, 3] = old_weight[:, 2].clone()
        new_conv.weight[:, 3] += torch.randn_like(
            new_conv.weight[:, 3]) * old_weight[:, 2].std() * 0.01
        if old_conv.bias is not None:
            new_conv.bias.data.copy_(old_conv.bias.data)
    model.model.model[0].conv = new_conv
    model.model.yaml['ch'] = 4

    cos = torch.nn.functional.cosine_similarity(
        new_conv.weight[:, 2].flatten().unsqueeze(0).float(),
        new_conv.weight[:, 3].flatten().unsqueeze(0).float()
    )
    print(f"[Fix] First conv: 3ch->4ch, Blue<->IR cosine={cos.item():.4f}")

    print("\n" + "=" * 60)
    if args.smoke:
        print("SMOKE TEST: 1 epoch (config check)")
    else:
        print("Exp B: YOLO26x 4ch + EQL v2 + small-image oversampling @1280")
    print("=" * 60)
    print(f"  Batch: {args.batch}, Workers: {args.workers}, Epochs: {args.epochs}")
    print(f"  Fixed 1280 (NO multi_scale) — proven eqlv2 recipe")
    print("=" * 60 + "\n")

    results = model.train(
        data=str(yaml_path),
        epochs=1 if args.smoke else args.epochs,
        imgsz=1280,
        batch=args.batch,
        device='cuda',
        workers=args.workers,
        amp=True,

        # SGD (NOT AdamW)
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        cos_lr=True,

        # Augmentation (identical to eqlv2 baseline)
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.05,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        erasing=0.1,

        # Training config
        close_mosaic=15,
        nbs=64,
        project=str(OUTPUT_DIR),
        name='train',
        exist_ok=True,
        save=not args.smoke,
        save_period=10,
        val=not args.smoke,
        plots=False,
        patience=30 if not args.smoke else 0,
        pretrained=True,
    )

    if not args.smoke:
        print("\n[OK] Exp B training complete!")
        print(f"Best model: {OUTPUT_DIR}/train/weights/best.pt")
    else:
        print("\n[OK] Smoke test complete")
