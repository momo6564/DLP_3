"""Plot helpers, so every figure in your report is labelled and readable."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files instead of opening windows

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PET_PALETTE = np.array([[31, 119, 180], [44, 160, 44], [214, 39, 40]], dtype=np.uint8)


def _save(fig, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("[fig] {}".format(out_path))
    return out_path


def plot_curves(history: dict, out_path, title: str = "") -> Path:
    """history = {"train_loss": [...], "val_loss": [...], "val_acc": [...]} - any subset."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key in ("train_loss", "val_loss"):
        if history.get(key):
            axes[0].plot(range(1, len(history[key]) + 1), history[key], marker="o", label=key)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for key in ("train_acc", "val_acc", "val_miou"):
        if history.get(key):
            axes[1].plot(range(1, len(history[key]) + 1), history[key], marker="o", label=key)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("metric")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title or "training curves")
    return _save(fig, out_path)


def plot_confusion_matrix(cm, class_names, out_path, normalize: bool = True,
                          title: str = "confusion matrix") -> Path:
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.5), max(4, n * 0.45)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, "{:.2f}".format(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out_path)


def overlay_mask(image, mask, alpha: float = 0.5, palette=PET_PALETTE):
    """image: HxWx3 uint8, mask: HxW integer class ids. Returns a blended uint8 image."""
    colour = palette[np.clip(mask, 0, len(palette) - 1)]
    return (image * (1 - alpha) + colour * alpha).astype(np.uint8)


def show_segmentation_grid(samples, out_path,
                           titles=("image", "ground truth", "prediction")) -> Path:
    """samples: list of (image_uint8_HWC, gt_mask_HW, pred_mask_HW). Keep it to 4-6 rows."""
    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    axes = np.atleast_2d(axes)
    for r, (image, gt, pred) in enumerate(samples):
        panels = (image, overlay_mask(image, gt), overlay_mask(image, pred))
        for c, panel in enumerate(panels):
            axes[r, c].imshow(panel)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c])
    return _save(fig, out_path)


def draw_boxes(image, boxes, labels=None, scores=None, class_names=None,
               colour=(255, 80, 0), thickness: int = 2):
    """boxes in absolute xyxy pixels. Returns a copy with boxes drawn (needs opencv)."""
    import cv2

    out = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
        parts = []
        if labels is not None and class_names is not None:
            parts.append(str(class_names[int(labels[i])]))
        elif labels is not None:
            parts.append(str(int(labels[i])))
        if scores is not None:
            parts.append("{:.2f}".format(float(scores[i])))
        if parts:
            cv2.putText(out, " ".join(parts), (x1, max(14, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return out


def plot_grad_norms(series: dict, out_path, title: str = "gradient reaching each timestep",
                    log_scale: bool = True) -> Path:
    """series = {"rnn": [...], "lstm": [...]} from utils.metrics.per_timestep_grad_norms.

    Position 0 is the first token. On a log scale a vanilla RNN's curve falls away from the
    end of the sequence while an LSTM's stays comparatively flat - that is the figure
    Question 3.4 asks for.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, values in series.items():
        ax.plot(range(len(values)), values, label=label, linewidth=1.8)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("timestep (0 = first token)")
    ax.set_ylabel("mean gradient L2 norm")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    return _save(fig, out_path)


def plot_length_study(results: dict, out_path, ylabel: str = "test accuracy",
                      title: str = "accuracy vs sequence length") -> Path:
    """results = {"rnn": {50: 0.78, 200: 0.74, 500: 0.62}, "lstm": {...}} for Question 3.3."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, mapping in results.items():
        lengths = sorted(mapping)
        ax.plot(lengths, [mapping[k] for k in lengths], marker="o", label=label)
    ax.set_xlabel("max sequence length (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def save_image(array, out_path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(array)
    ax.axis("off")
    return _save(fig, out_path)
