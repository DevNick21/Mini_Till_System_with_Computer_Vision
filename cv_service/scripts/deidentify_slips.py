"""
Deidentify slip writer folders by renaming to anon_### and write a reversible CSV mapping.

Usage:
  python -m cv_service.scripts.deidentify_slips \
    --src-dir cv_service/slips \
    --out-dir cv_service/slips_anon \
    --mapping-file cv_service/slips_deid_mapping.csv

Notes:
  - Only top-level folders under src-dir are considered writers.
  - Images are copied (not moved) to out-dir to keep originals intact.
  - Mapping CSV columns: anon_id,original_name
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_writers(src_dir: Path) -> list[Path]:
    return sorted([p for p in src_dir.iterdir() if p.is_dir()])


def ensure_empty_dir(dir_path: Path):
    if dir_path.exists():
        # Keep safe: if not empty and not explicitly allowed, raise
        if any(dir_path.iterdir()):
            raise RuntimeError(f"Output directory is not empty: {dir_path}")
    else:
        dir_path.mkdir(parents=True, exist_ok=True)


def write_mapping(mapping_path: Path, names: list[str]):
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["anon_id", "original_name"])  # header
        for idx, name in enumerate(names, start=1):
            w.writerow([f"anon_{idx:03d}", name])


def copy_images(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in src.glob("**/*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            target = dst / p.name
            # If name collision, add index suffix
            if target.exists():
                stem, ext = target.stem, target.suffix
                k = 1
                while True:
                    alt = dst / f"{stem}_{k}{ext}"
                    if not alt.exists():
                        target = alt
                        break
                    k += 1
            shutil.copy2(p, target)
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, default=Path("cv_service/slips"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("cv_service/slips_anon"))
    ap.add_argument("--mapping-file", type=Path,
                    default=Path("cv_service/slips_deid_mapping.csv"))
    args = ap.parse_args()

    writers = discover_writers(args.src_dir)
    if not writers:
        raise SystemExit(f"No writer folders found under {args.src_dir}")

    ensure_empty_dir(args.out_dir)

    # Stable order by folder name
    original_names = [p.name for p in writers]
    write_mapping(args.mapping_file, original_names)

    total = 0
    for idx, writer_path in enumerate(writers, start=1):
        anon_name = f"anon_{idx:03d}"
        dst = args.out_dir / anon_name
        copied = copy_images(writer_path, dst)
        print(f"{writer_path.name:>12} -> {anon_name}: {copied}")
        total += copied

    print(f"Done. Copied {total} images across {len(writers)} writers.")
    print(f"Mapping written to: {args.mapping_file}")


if __name__ == "__main__":
    main()
