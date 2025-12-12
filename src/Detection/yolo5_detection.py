from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_pairs(
    image_dir: Path, label_dir: Path
) -> List[Tuple[Path, Path]]:
    images: Dict[str, Path] = {
        p.stem: p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    }
    labels: Dict[str, Path] = {
        p.stem: p for p in label_dir.glob("*.txt") if p.is_file()
    }

    missing_images = sorted(labels.keys() - images.keys())
    missing_labels = sorted(images.keys() - labels.keys())
    if missing_images:
        print(f"Skipping {len(missing_images)} label(s) with no matching image")
    if missing_labels:
        print(f"Skipping {len(missing_labels)} image(s) with no matching label")

    common = sorted(images.keys() & labels.keys())
    if not common:
        raise ValueError("No matching image/label pairs found.")

    return [(images[k], labels[k]) for k in common]


def split_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[Path, Path]]]:
    if not (0 <= val_ratio < 1 and 0 <= test_ratio < 1):
        raise ValueError("Ratios must be between 0 and 1.")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1.")

    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    total = len(pairs)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)

    splits: Dict[str, List[Tuple[Path, Path]]] = {}
    offset = 0
    if test_count > 0:
        splits["test"] = pairs[offset:offset + test_count]
        offset += test_count
    splits["val"] = pairs[offset:offset + val_count]
    splits["train"] = pairs[offset + val_count:]
    return splits


def copy_subset(
    subset: Iterable[Tuple[Path, Path]],
    out_dir: Path,
    split_name: str,
) -> None:
    img_dst = out_dir / split_name / "images"
    lbl_dst = out_dir / split_name / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in subset:
        shutil.copy2(img_path, img_dst / img_path.name)
        shutil.copy2(lbl_path, lbl_dst / lbl_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split image/label pairs into train/val/test sets."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("/home/panda/projects/german-street-sign/Data/raw_data/mask_train_data/img"),
        help="Directory that contains the source images.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("/home/panda/projects/german-street-sign/Data/raw_data/mask_train_data/label"),
        help="Directory that contains the source label files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/panda/projects/german-street-sign/Data/processed_data/mask_split"),
        help="Destination directory for the split dataset.",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Fraction of samples for validation."
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Fraction of samples for testing (set >0 only if a test split is needed).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible splits."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = collect_pairs(args.images, args.labels)
    splits = split_pairs(pairs, args.val_ratio, args.test_ratio, args.seed)

    for split_name, subset in splits.items():
        if subset:
            copy_subset(subset, args.output, split_name)
            print(f"{split_name}: {len(subset)} pairs")
        else:
            print(f"{split_name}: 0 pairs (skipped)")

    print(f"Finished writing data to {args.output}")


if __name__ == "__main__":
    main()
