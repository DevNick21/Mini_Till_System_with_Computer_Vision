"""
Stratified split of per-writer dataset into train/val/test folders.

Usage:
  python -m cv_service.scripts.split_dataset \
    --data-dir cv_service/slips_anon \
    --out-dir cv_service/dataset_splits \
    --train 0.7 --val 0.2 --test 0.1 \
    --seed 42 --min-per-writer 10

Rules:
  - Keeps directory names under each split.
  - Ensures each writer has at least min-per-writer images; otherwise fails.
  - Uses random but reproducible shuffling with --seed.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
import shutil

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(dir_path: Path):
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("cv_service/slips"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("cv_service/dataset_splits"))
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-per-writer", type=int, default=6)
    args = ap.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise SystemExit("Splits must sum to 1.0")

    random.seed(args.seed)

    writers = [p for p in args.data_dir.iterdir() if p.is_dir()]
    if not writers:
        raise SystemExit(f"No writer folders in {args.data_dir}")

    # Prepare out dirs
    for split in ("train", "val", "test"):
        safe_mkdir(args.out_dir / split)

    total = {"train": 0, "val": 0, "test": 0}

    for wdir in sorted(writers):
        images = list_images(wdir)
        if len(images) < args.min_per_writer:
            raise SystemExit(
                f"Writer {wdir.name} has only {len(images)} images (< {args.min_per_writer})")

        random.shuffle(images)
        n = len(images)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        n_test = n - n_train - n_val

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, files in splits.items():
            out_wdir = args.out_dir / split / wdir.name
            safe_mkdir(out_wdir)
            for src in files:
                dst = out_wdir / src.name
                shutil.copy2(src, dst)
            total[split] += len(files)

        print(
            f"{wdir.name}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    print(
        f"Done. train={total['train']} val={total['val']} test={total['test']}")


if __name__ == "__main__":
    main()
