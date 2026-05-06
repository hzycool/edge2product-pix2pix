import argparse
import json
from pathlib import Path

from PIL import Image

from utils.metrics import load_rgb, mean_l1, psnr, ssim


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated images against targets.")
    parser.add_argument("--generated_dir", required=True)
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--save_path", required=True)
    return parser.parse_args()


def find_images(directory):
    directory = Path(directory)
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)


def resize_target_if_needed(target_path, reference_path):
    ref_size = Image.open(reference_path).size
    target = Image.open(target_path).convert("RGB")
    if target.size == ref_size:
        return target_path
    resized = target.resize(ref_size, Image.BICUBIC)
    temp_path = target_path.parent / f".resized_{target_path.name}"
    resized.save(temp_path)
    return temp_path


def main():
    args = parse_args()
    generated_paths = find_images(args.generated_dir)
    target_paths = find_images(args.target_dir)
    target_by_name = {p.name: p for p in target_paths}
    target_by_stem = {p.stem: p for p in target_paths}

    l1_values, psnr_values, ssim_values = [], [], []
    warnings = []
    for generated_path in generated_paths:
        target_path = target_by_name.get(generated_path.name) or target_by_stem.get(generated_path.stem)
        if target_path is None:
            warnings.append(f"WARNING: no target found for {generated_path.name}; skipped.")
            continue
        if Image.open(generated_path).size != Image.open(target_path).size:
            warnings.append(f"WARNING: size mismatch for {generated_path.name}; target resized for evaluation.")
            target_path = resize_target_if_needed(target_path, generated_path)
        generated = load_rgb(generated_path)
        target = load_rgb(target_path)
        l1_values.append(mean_l1(generated, target))
        psnr_values.append(psnr(generated, target))
        ssim_values.append(ssim(generated, target))

    if not l1_values:
        raise RuntimeError("No matched image pairs were found for evaluation.")

    metrics = {
        "num_pairs": len(l1_values),
        "mean_l1": sum(l1_values) / len(l1_values),
        "psnr": sum(psnr_values) / len(psnr_values),
        "ssim": sum(ssim_values) / len(ssim_values),
        "warnings": warnings,
    }
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    md_path = save_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Metrics\n\n")
        f.write(f"- Matched pairs: {metrics['num_pairs']}\n")
        f.write(f"- Mean L1: {metrics['mean_l1']:.6f}\n")
        f.write(f"- PSNR: {metrics['psnr']:.4f}\n")
        f.write(f"- SSIM: {metrics['ssim']:.4f}\n")
        if warnings:
            f.write("\n## Warnings\n\n")
            for warning in warnings:
                f.write(f"- {warning}\n")

    for warning in warnings:
        print(warning)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved metrics to {save_path} and {md_path}")


if __name__ == "__main__":
    main()
