"""Download every dataset the assignment needs. Run once, then go offline if you like.

    python setup/download_datasets.py --all
    python setup/download_datasets.py --voc     # Q1, object detection
    python setup/download_datasets.py --pet     # Q2, semantic segmentation
    python setup/download_datasets.py --imdb    # Q3, RNNs and LSTMs

Disk budget: PASCAL VOC 2007 ~900 MB, Oxford-IIIT Pet ~800 MB, IMDB ~130 MB.
All three are standard academic benchmarks; there is nothing to sign up for and nothing
to scrape.
"""
from __future__ import annotations

import argparse
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
IMDB_MD5 = "7c2ac02c03563afcf9b574c7e56c153a"


def download_voc(root: Path) -> None:
    """PASCAL VOC 2007 detection: 20 classes, ~5k trainval + ~5k test images. Q1."""
    from torchvision.datasets import VOCDetection

    for image_set in ("trainval", "test"):
        VOCDetection(root=str(root), year="2007", image_set=image_set, download=True)
    print(f"[ok] PASCAL VOC 2007 ready under {root / 'VOCdevkit'}")
    print("     Next: python setup/voc_to_yolo.py   (builds the shared detection split)")


def download_pet(root: Path) -> None:
    """Oxford-IIIT Pet: 7,349 images with pixel-level trimaps. Q2."""
    from torchvision.datasets import OxfordIIITPet

    for split in ("trainval", "test"):
        OxfordIIITPet(root=str(root), split=split, target_types="segmentation", download=True)
    print(f"[ok] Oxford-IIIT Pet ready under {root / 'oxford-iiit-pet'}")


def download_imdb(root: Path) -> None:
    """Large Movie Review Dataset (Maas et al., 2011): 25k train / 25k test, Q3.

    Plain text in folders, so there is no tokenizer to install - utils/text.py handles it.
    """
    from torchvision.datasets.utils import download_and_extract_archive

    target = root / "aclImdb"
    if target.is_dir():
        print(f"[ok] IMDB already present at {target}")
        return
    download_and_extract_archive(IMDB_URL, download_root=str(root), md5=IMDB_MD5)
    print(f"[ok] IMDB ready under {target}")
    print("     25,000 labelled training reviews, 25,000 test, balanced pos/neg.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--voc", action="store_true", help="Q1 - detection")
    ap.add_argument("--pet", action="store_true", help="Q2 - segmentation")
    ap.add_argument("--imdb", action="store_true", help="Q3 - RNN / LSTM")
    ap.add_argument("--root", default=str(DATA_ROOT))
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if not any([args.all, args.voc, args.pet, args.imdb]):
        ap.error("pick at least one of --all / --voc / --pet / --imdb")

    if args.all or args.voc:
        download_voc(root)
    if args.all or args.pet:
        download_pet(root)
    if args.all or args.imdb:
        download_imdb(root)

    print("\nIf a download stalls or a mirror 404s, see the Troubleshooting section in")
    print("README.md - there are documented fallbacks for VOC and IMDB.")


if __name__ == "__main__":
    main()
