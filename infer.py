import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import PairedImageDataset
from models import GeneratorUNet
from utils.checkpoint import load_generator
from utils.visualize import save_comparison_grid, tensor_to_pil


def parse_args():
    parser = argparse.ArgumentParser(description="Run Pix2Pix inference.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save_dir", default="./outputs/edge2shoes_100/inference")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--direction", choices=["AtoB", "BtoA"], default="AtoB")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_images", type=int, default=20)
    return parser.parse_args()


def choose_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return requested


def main():
    args = parse_args()
    device = choose_device(args.device)
    save_dir = Path(args.save_dir)
    generated_dir = save_dir / "generated"
    input_dir = save_dir / "input"
    target_dir = save_dir / "target"
    for directory in (generated_dir, input_dir, target_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset = PairedImageDataset(
        args.dataroot,
        split="test",
        img_size=args.img_size,
        direction=args.direction,
        sample_size=args.num_images,
        allow_unpaired=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    generator = GeneratorUNet().to(device)
    load_generator(generator, args.checkpoint, device=device)
    generator.eval()

    collected_inputs, collected_targets, collected_outputs = [], [], []
    has_target = False
    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= args.num_images:
                break
            real_A = batch["input"].to(device)
            fake_B = generator(real_A)
            real_B = batch.get("target")
            name = f"sample_{count:03d}.png"

            tensor_to_pil(real_A.cpu()).save(input_dir / name)
            tensor_to_pil(fake_B.cpu()).save(generated_dir / name)
            if real_B is not None:
                has_target = True
                tensor_to_pil(real_B).save(target_dir / name)
                collected_targets.append(real_B)
            collected_inputs.append(real_A.cpu())
            collected_outputs.append(fake_B.cpu())
            count += 1

    if count == 0:
        raise RuntimeError("No images were processed during inference.")

    inputs = torch.cat(collected_inputs, dim=0)
    outputs = torch.cat(collected_outputs, dim=0)
    targets = torch.cat(collected_targets, dim=0) if has_target else None
    labels = ["input", "target", "generated"] if has_target else ["input", "generated"]
    save_comparison_grid(inputs, targets, outputs, save_dir / "inference_grid.png", max_images=min(count, 8), labels=labels)
    print(f"Saved {count} generated images to {generated_dir}")
    print(f"Saved comparison grid to {save_dir / 'inference_grid.png'}")


if __name__ == "__main__":
    main()
