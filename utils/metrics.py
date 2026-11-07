"""Metrics you may reuse: segmentation IoU / Dice, latency, model size, gradient flow.

Everything here is standard and deliberately boring. Marks are awarded for the
experiments and the analysis, not for re-deriving a confusion matrix by hand.
"""
from __future__ import annotations

import time

import numpy as np
import torch


class SegmentationMeter:
    """Streaming confusion matrix for semantic segmentation (Question 2).

    Usage:
        meter = SegmentationMeter(num_classes=3)
        for images, masks in loader:
            logits = model(images)
            meter.update(logits.argmax(1), masks)
        print(meter.summary(["pet", "background", "boundary"]))
    """

    def __init__(self, num_classes: int, ignore_index=None) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, preds, targets) -> None:
        preds = preds.detach().flatten().cpu().numpy()
        targets = targets.detach().flatten().cpu().numpy()
        valid = (targets >= 0) & (targets < self.num_classes)
        if self.ignore_index is not None:
            valid &= targets != self.ignore_index
        idx = targets[valid] * self.num_classes + preds[valid]
        self.confusion += np.bincount(
            idx, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)

    def per_class_iou(self):
        cm = self.confusion.astype(np.float64)
        intersection = np.diag(cm)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(union > 0, intersection / union, np.nan)

    def per_class_dice(self):
        cm = self.confusion.astype(np.float64)
        intersection = np.diag(cm)
        denom = cm.sum(axis=1) + cm.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 0, 2 * intersection / denom, np.nan)

    def pixel_accuracy(self) -> float:
        cm = self.confusion
        return float(np.diag(cm).sum() / max(cm.sum(), 1))

    def summary(self, class_names=None) -> dict:
        iou = self.per_class_iou()
        dice = self.per_class_dice()
        names = class_names or ["class_{}".format(i) for i in range(self.num_classes)]
        return {
            "pixel_accuracy": round(self.pixel_accuracy(), 4),
            "mIoU": round(float(np.nanmean(iou)), 4),
            "mDice": round(float(np.nanmean(dice)), 4),
            "per_class_iou": {n: round(float(v), 4) for n, v in zip(names, iou)},
            "per_class_dice": {n: round(float(v), 4) for n, v in zip(names, dice)},
        }


@torch.no_grad()
def benchmark_latency(model, input_shape=(1, 3, 640, 640), device=None,
                      warmup: int = 10, iters: int = 50) -> dict:
    """Single-image latency and FPS, needed for the Q1 speed / accuracy trade-off table.

    Always state the device you measured on. A CPU number and a T4 number are not
    comparable, and a reviewer will notice if you mix them in one table.
    """
    device = device or next(model.parameters()).device
    model.eval().to(device)
    dummy = torch.randn(*input_shape, device=device)

    for _ in range(warmup):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms = 1000.0 * elapsed / iters
    return {"device": str(device), "latency_ms": round(ms, 2), "fps": round(1000.0 / ms, 1)}


def model_size_mb(model) -> float:
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return round((params + buffers) / (1024.0 ** 2), 2)


def per_timestep_grad_norms(model, batch, loss_fn, device, embed_attr: str = "embedding"):
    """Gradient norm reaching each timestep of a sequence (Question 3.4).

    This is the measurement behind the phrase 'vanishing gradient'. It hooks the embedding
    output, runs one forward/backward pass, and returns the mean L2 norm of the gradient
    arriving at each position t.

    A vanilla RNN typically shows norms that decay by orders of magnitude as you move away
    from the end of the sequence; an LSTM keeps them far flatter, because the cell state
    carries gradient along an additive path. Plot both on a log scale.

    Returns a list of length T (position 0 = first token).
    """
    model.to(device).train()
    tokens, lengths, labels = batch
    tokens, labels = tokens.to(device), labels.to(device)

    captured = {}

    def hook(module, inputs, output):
        output.retain_grad()
        captured["embedded"] = output

    handle = getattr(model, embed_attr).register_forward_hook(hook)
    try:
        logits = model(tokens, lengths)
        loss = loss_fn(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()
    finally:
        handle.remove()

    grad = captured["embedded"].grad          # (batch, time, embed_dim)
    if grad is None:
        raise RuntimeError(
            "no gradient captured - check that model.{} is the embedding layer and that "
            "the batch reached it".format(embed_attr))
    norms = grad.norm(dim=2).mean(dim=0)      # (time,)
    model.zero_grad(set_to_none=True)
    return [float(v) for v in norms.detach().cpu()]
