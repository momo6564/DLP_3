"""Run this FIRST. It tells you whether your environment can finish the assignment.

    python env_check.py
"""
import importlib
import platform
import sys

PACKAGES = [
    ("torch", "core framework (also gives you nn.RNN / nn.LSTM for Q3)"),
    ("torchvision", "pretrained detectors + datasets"),
    ("ultralytics", "Q1a - YOLOv8"),
    ("segmentation_models_pytorch", "Q2 - U-Net with pretrained encoder"),
    ("torchmetrics", "Q1/Q2 metrics"),
    ("pycocotools", "Q1b - mAP for Faster R-CNN"),
    ("cv2", "image I/O and drawing"),
    ("matplotlib", "plots"),
    ("sklearn", "confusion matrix, reports"),
    ("pandas", "results tables"),
    ("tqdm", "progress bars"),
]


def main() -> int:
    print(f"Python  : {sys.version.split()[0]}  ({platform.platform()})")
    missing = []
    for name, why in PACKAGES:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - we want the reason printed
            missing.append(name)
            print(f"[MISSING] {name:32s} {why}  ({type(exc).__name__})")
        else:
            version = getattr(mod, "__version__", "?")
            print(f"[  OK   ] {name:32s} {version}")

    try:
        import torch

        print("\nCUDA available :", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU            :", torch.cuda.get_device_name(0))
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU memory     : {total:.1f} GiB")
        else:
            print("No GPU found. Use Google Colab (Runtime > Change runtime type > T4 GPU),")
            print("or shrink the image size / subset sizes in config.yaml.")
            print("Q3 (RNN/LSTM) is the one question that is comfortable on CPU.")
    except Exception:
        print("\ntorch not importable - fix that before anything else.")

    if missing:
        print("\nInstall the missing pieces with:")
        print("   pip install " + " ".join(
            {"cv2": "opencv-python", "sklearn": "scikit-learn"}.get(m, m) for m in missing
        ))
        return 1
    print("\nEnvironment looks good. Start with setup/download_datasets.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
