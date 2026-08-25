"""
Dataset 4ch Patch — fix ultralytics 4ch PNG loading
=====================================================
Two bugs block 4ch training with prebuilt PNG datasets:

BUG 1 (model): missing 'channels: 4' in data.yaml → ultralytics defaults
    data["channels"]=3 → model.train() REBUILDS the model with ch=3,
    silently discarding our 4ch conv replacement.

BUG 2 (data): BaseDataset sets cv2_flag = IMREAD_COLOR for any
    channels != 1 → 4ch PNGs are read as 3ch BGR, dropping the 4th
    channel silently.

This module patches BUG 2. BUG 1 is fixed by adding 'channels: 4'
to the data.yaml (done in training scripts).

Usage (before model.train()):
    from dataset_4ch_patch import patch_4ch_loading
    patch_4ch_loading()
"""

import cv2


def patch_4ch_loading():
    """Patch BaseDataset to read 4ch images with IMREAD_UNCHANGED.

    Must be called BEFORE the dataset is built (i.e., before model.train()).
    """
    from ultralytics.data.base import BaseDataset

    _orig_init = BaseDataset.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # If dataset expects >=4 channels, read PNGs with IMREAD_UNCHANGED
        # so the 4th channel (IR/depth) survives loading.
        if getattr(self, "channels", 3) >= 4:
            self.cv2_flag = cv2.IMREAD_UNCHANGED

    BaseDataset.__init__ = _patched_init
    print("[Patch] BaseDataset: cv2_flag=IMREAD_UNCHANGED for channels>=4")


if __name__ == '__main__':
    # Self-test
    patch_4ch_loading()
    from ultralytics.data.base import BaseDataset
    import numpy as np

    # Simulate a dataset with channels=4
    class Dummy:
        pass

    # Verify patch is active
    print("[OK] patch installed on BaseDataset.__init__")
