"""Convert PASCAL VOC 2007 into the YOLO layout Ultralytics expects (Question 1).

You are NOT asked to write this conversion yourself - that is plumbing, not learning.
Run it once after download_datasets.py:

    python setup/voc_to_yolo.py                      # uses classes from config.yaml
    python setup/voc_to_yolo.py --classes person car dog

Output (data/voc_yolo):
    images/train/*.jpg   labels/train/*.txt
    images/val/*.jpg     labels/val/*.txt
    data.yaml

One label line per object:  <class_id> <x_center> <y_center> <width> <height>
all four box values normalised to [0, 1] by image width/height.

The same filtered split is reused by the Faster R-CNN script (starter/q1b_faster_rcnn.py),
so both detectors are trained and evaluated on exactly the same images. Do not regenerate
the split between the two runs or your comparison stops being fair.
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_cfg() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def voc_boxes(xml_path: Path, keep: list, skip_difficult: bool = True):
    """Return (width, height, [(class_name, xmin, ymin, xmax, ymax), ...])."""
    tree = ET.parse(xml_path)
    size = tree.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))

    objects = []
    for obj in tree.findall("object"):
        name = obj.findtext("name")
        if name not in keep:
            continue
        if skip_difficult and obj.findtext("difficult") == "1":
            continue
        bb = obj.find("bndbox")
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))
        if xmax <= xmin or ymax <= ymin:
            continue
        objects.append((name, xmin, ymin, xmax, ymax))
    return width, height, objects


def to_yolo_line(class_id: int, box, width: int, height: int) -> str:
    _, xmin, ymin, xmax, ymax = box
    xc = ((xmin + xmax) / 2) / width
    yc = ((ymin + ymax) / 2) / height
    bw = (xmax - xmin) / width
    bh = (ymax - ymin) / height
    xc, yc, bw, bh = (min(max(v, 0.0), 1.0) for v in (xc, yc, bw, bh))
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    cfg = load_cfg()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voc-root", default=str(ROOT / "data" / "VOCdevkit" / "VOC2007"))
    ap.add_argument("--out", default=str(ROOT / "data" / "voc_yolo"))
    ap.add_argument("--classes", nargs="+", default=cfg["q1_detection"]["classes"])
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--max-images", type=int,
                    default=cfg["q1_detection"].get("max_images", 2500),
                    help="cap the dataset size (0 = no cap, which is about 5-6k images). The "
                         "default comes from config.yaml and keeps each detector near 25 min "
                         "on a T4. Whatever you use, keep it the same for both detectors.")
    ap.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    args = ap.parse_args()

    voc = Path(args.voc_root)
    ann_dir, img_dir = voc / "Annotations", voc / "JPEGImages"
    if not ann_dir.is_dir():
        raise SystemExit(f"{ann_dir} not found. Run: python setup/download_datasets.py --voc")

    keep = list(args.classes)
    class_to_id = {name: i for i, name in enumerate(keep)}

    ids = [p.stem for p in sorted(ann_dir.glob("*.xml"))]
    kept = []
    for image_id in ids:
        width, height, objects = voc_boxes(ann_dir / f"{image_id}.xml", keep)
        if objects:                      # drop images with none of our classes
            kept.append((image_id, width, height, objects))

    random.Random(args.seed).shuffle(kept)
    if args.max_images:
        kept = kept[: args.max_images]

    n_val = max(1, int(len(kept) * args.val_fraction))
    splits = {"val": kept[:n_val], "train": kept[n_val:]}

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in splits:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in keep}
    for split, rows in splits.items():
        for image_id, width, height, objects in rows:
            shutil.copy2(img_dir / f"{image_id}.jpg", out / "images" / split / f"{image_id}.jpg")
            lines = []
            for box in objects:
                counts[box[0]] += 1
                lines.append(to_yolo_line(class_to_id[box[0]], box, width, height))
            (out / "labels" / split / f"{image_id}.txt").write_text("\n".join(lines) + "\n",
                                                                    encoding="utf-8")

    data_yaml = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for name, i in class_to_id.items()},
    }
    with open(out / "data.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(data_yaml, fh, sort_keys=False)

    print(f"train images : {len(splits['train'])}")
    print(f"val images   : {len(splits['val'])}")
    print("objects per class (whole set):")
    for name, n in counts.items():
        print(f"   {name:10s} {n}")
    print(f"\ndata.yaml written to {out / 'data.yaml'}")
    print("Copy the per-class counts into your report - the class imbalance here explains")
    print("a lot about the per-class AP numbers you are about to see.")


if __name__ == "__main__":
    main()
