"""
EQL v2 (Equalization Loss v2) for YOLO/Object Detection
========================================================
Paper: "Equalization Loss v2: A New Gradient Balance Approach for
Long-tailed Object Detection" — Tan et al., CVPR 2021

Correct mechanism (from the paper):
  g_j = accumulated |positive gradient| / |negative gradient| for class j
  f(g) = sigmoid(γ * (g - μ))   ← maps gradient ratio to [0, 1]

  q_j = 1 + α * (1 - f(g_j))    ← POSITIVE gradient BOOST (rare classes: g→0, q→1+α)
  r_j = f(g_j)                  ← NEGATIVE gradient SUPPRESSION (rare classes: g→0, r→0)

Effect:
  - Rare classes: little positive gradient naturally → g_j ≈ 0 → r_j ≈ 0
    → negative gradient SUPPRESSED → model less biased against predicting them
  - Common classes: lots of positive gradient → g_j ≈ 1 → r_j ≈ 1
    → negative gradient normal → no change

Integration: replaces nn.BCEWithLogitsLoss in ultralytics' v8DetectionLoss.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class EQLv2BCELoss(nn.Module):
    """BCE loss with EQL v2 per-class gradient reweighting.

    Weights positive and negative contributions independently per class:
      loss[j] = q_j * BCE_pos + r_j * BCE_neg

    where q_j boosts rare-class positive gradient, r_j suppresses rare-class
    negative gradient.
    """

    def __init__(
        self,
        num_classes: int,
        class_counts: List[int],
        alpha: float = 0.5,
        gamma: float = 4.0,
        mu: float = 0.5,
    ):
        """
        Args:
            num_classes: number of detection classes (excluding background)
            class_counts: per-class instance counts in training set
            alpha: max positive boost factor (default 0.5 → max q=1.5)
            gamma: sigmoid steepness (higher = sharper transition)
            mu: sigmoid center (shift left=more classes treated as rare)
        """
        super().__init__()
        self.nc = num_classes
        self.alpha = alpha
        self.gamma_s = gamma  # rename to avoid conflict
        self.mu_val = mu

        # Compute initial gradient ratio g_j from class frequency
        # g_j ≈ freq_j / (1 - freq_j): ratio of positive to negative samples
        total = sum(class_counts)
        freq = torch.tensor([max(c, 1) / total for c in class_counts],
                            dtype=torch.float32)
        g_init = freq / (1.0 - freq + 1e-8)  # avoid div by zero

        # Register buffers (saved with model state)
        self.register_buffer('g', g_init)
        self.register_buffer('q', torch.ones(num_classes))
        self.register_buffer('r', torch.ones(num_classes))
        self.update_weights()

    def update_weights(self):
        """Recompute q, r weights from current gradient ratio g."""
        # f(g) = sigmoid(γ * (g - μ))
        f = torch.sigmoid(self.gamma_s * (self.g.to(self.g.device) - self.mu_val))

        # q = 1 + α * (1 - f)  → positive boost for rare classes (g→0 → q→1+α)
        q_new = 1.0 + self.alpha * (1.0 - f)

        # r = f  → negative suppression for rare classes (g→0 → r→0)
        r_new = f

        # Clamp to safe ranges
        self.q.copy_(torch.clamp(q_new, 1.0, 1.0 + self.alpha))
        self.r.copy_(torch.clamp(r_new, 1e-6, 1.0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute EQL v2 weighted BCE loss.

        Args:
            pred: (bs, num_anchors, nc) logits
            target: (bs, num_anchors, nc) soft targets (0 for negatives, >0 for positives)

        Returns:
            loss: (bs, num_anchors, nc) weighted BCE loss
        """
        # Standard BCE loss (reduction='none')
        loss = nn.functional.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        )  # (bs, anchors, nc)

        # Get weights as (1, 1, nc) for broadcasting
        q = self.q.to(pred.device).view(1, 1, -1)
        r = self.r.to(pred.device).view(1, 1, -1)

        # Identify positive vs negative targets
        # target > 0: positives (assigned anchors for that class)
        # target == 0: negatives (background or other classes)
        pos_mask = target > 0
        neg_mask = ~pos_mask

        # Apply EQL v2 weights
        # Positive samples: weight by q (boost for rare classes)
        # Negative samples: weight by r (suppress for rare classes)
        loss = torch.where(pos_mask, loss * q, loss * r)

        return loss

    def extra_repr(self) -> str:
        return (f"nc={self.nc}, alpha={self.alpha}, gamma={self.gamma_s}, "
                f"mu={self.mu_val}")


# ============================================================
# Ultralytics Integration
# ============================================================

_EQL_CONFIG = None  # global config set by install function


def install_eql_v2_hook(
    class_counts: List[int],
    alpha: float = 0.5,
    gamma: float = 4.0,
    mu: float = 0.5,
):
    """Install EQL v2 loss hook into ultralytics training pipeline.

    Monkey-patches v8DetectionLoss.__init__ to replace nn.BCEWithLogitsLoss
    with EQLv2BCELoss. The patch is transparent — no changes needed to
    training scripts.

    Args:
        class_counts: per-class training instance counts [n0, n1, ..., n11]
        alpha: positive gradient boost factor (0.5 = max 50% boost for rare)
        gamma: sigmoid steepness (4.0 = default, higher = sharper threshold)
        mu: sigmoid center (0.5 = default, lower = more classes treated as rare)

    Returns:
        EQLv2BCELoss instance (for inspection)
    """
    global _EQL_CONFIG
    from ultralytics.utils.loss import v8DetectionLoss

    _EQL_CONFIG = {
        'nc': len(class_counts),
        'class_counts': class_counts,
        'alpha': alpha,
        'gamma': gamma,
        'mu': mu,
    }

    # Store original __init__
    _original_init = v8DetectionLoss.__init__

    def _patched_init(self, model, *args, **kwargs):
        """Patched __init__: replace self.bce with EQLv2BCELoss."""
        # Call original init first
        _original_init(self, model, *args, **kwargs)

        # Disable class_weights (EQL v2 handles weighting internally)
        self.class_weights = None

        # Replace BCE with EQL v2
        eql_bce = EQLv2BCELoss(
            num_classes=_EQL_CONFIG['nc'],
            class_counts=_EQL_CONFIG['class_counts'],
            alpha=_EQL_CONFIG['alpha'],
            gamma=_EQL_CONFIG['gamma'],
            mu=_EQL_CONFIG['mu'],
        )
        self.bce = eql_bce
        self._eql_v2 = eql_bce  # reference for inspection

    v8DetectionLoss.__init__ = _patched_init

    # Also need to patch E2ELoss for YOLO26 (end2end models)
    # E2ELoss wraps v8DetectionLoss, so the patch flows through automatically
    # since E2ELoss.__init__ calls v8DetectionLoss(model, ...)

    print(f"\n[EQL v2] Loss hook installed")
    print(f"  Classes: {len(class_counts)}, alpha={alpha}, gamma={gamma}, mu={mu}")

    # Compute and display initial weights
    total = sum(class_counts)
    freq = [c / total for c in class_counts]
    g_init = [f / (1 - f + 1e-8) for f in freq]
    f_vals = [1.0 / (1.0 + torch.exp(torch.tensor(-gamma * (g - mu))).item())
              for g in g_init]
    q_vals = [1.0 + alpha * (1.0 - f) for f in f_vals]
    r_vals = f_vals

    print(f"  Initial weights (static, from class frequency):")
    print(f"  {'Class':<8} {'Count':>6} {'Freq':>8} {'g_init':>8} "
          f"{'q(pos)':>8} {'r(neg)':>8}")
    for i in range(len(class_counts)):
        print(f"  {i:<8} {class_counts[i]:>6} {freq[i]:>7.4f} "
              f"{g_init[i]:>8.4f} {q_vals[i]:>8.4f} {r_vals[i]:>8.4f}")
    print(f"  Rare classes (r<0.5): ", end="")
    for i, r in enumerate(r_vals):
        if r < 0.5:
            print(f"class_{i}(r={r:.3f}) ", end="")
    print()

    return None  # patch is global, no instance needed


def uninstall_eql_v2_hook():
    """Restore original v8DetectionLoss.__init__."""
    from ultralytics.utils.loss import v8DetectionLoss
    global _EQL_CONFIG
    _EQL_CONFIG = None
    if hasattr(v8DetectionLoss, '_original_init'):
        v8DetectionLoss.__init__ = v8DetectionLoss._original_init
        print("[EQL v2] Hook uninstalled")


if __name__ == '__main__':
    # Demo
    CLASS_COUNTS = [5499, 135, 3061, 621, 809, 666, 1571, 89, 1455, 282, 204, 25]

    print("=" * 60)
    print("EQL v2 Weight Demo")
    print("=" * 60)

    eql = EQLv2BCELoss(
        num_classes=12,
        class_counts=CLASS_COUNTS,
        alpha=0.5,
        gamma=4.0,
        mu=0.5,
    )

    print(f"\nInitial positive weights (q): {eql.q.tolist()}")
    print(f"Initial negative weights (r): {eql.r.tolist()}")
    print(f"\nEffect summary:")
    print(f"  class_0  (n=5499): pos={eql.q[0].item():.3f}x, neg={eql.r[0].item():.3f}x → normal")
    print(f"  class_7  (n=89):   pos={eql.q[7].item():.3f}x, neg={eql.r[7].item():.3f}x → neg suppressed!")
    print(f"  class_11 (n=25):   pos={eql.q[11].item():.3f}x, neg={eql.r[11].item():.3f}x → neg heavily suppressed!")
    print(f"\n  class_11 negative gradient reduced to {eql.r[11].item()*100:.1f}%")
    print(f"  → model {1/eql.r[11].item():.0f}x less penalized for false positives on class_11")
    print(f"  → allows ~{1/eql.r[11].item():.0f}x more detections before reaching same loss")
