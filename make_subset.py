import argparse
import random
import shutil
from pathlib import Path


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a small reproducible subset from edges2shoes.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_images(directory):
    return sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)


def copy_split(src_dir, dst_dir, sample_size, seed):
    images = list_images(src_dir)
    if not images:
        return 0
    rng = random.Random(seed)
    selected = images if len(images) <= sample_size else rng.sample(images, sample_size)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in selected:
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return len(selected)


def main():
    args = parse_args()
    src_root = Path(args.dataroot)
    output_root = Path(args.output_root)
    if not src_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {src_root}")

    split_dirs = [name for name in ("train", "val", "test") if (src_root / name).exists()]
    counts = {}
    if split_dirs:
        for split in split_dirs:
            size = args.sample_size if split == "train" else min(max(10, args.sample_size // 5), args.sample_size)
            counts[split] = copy_split(src_root / split, output_root / split, size, args.seed)
    else:
        counts["train"] = copy_split(src_root, output_root / "train", args.sample_size, args.seed)

    print(f"Created subset at {output_root}")
    for split, count in counts.items():
        print(f"- {split}: {count} images")


if __name__ == "__main__":
    main()
