import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import PairedImageDataset
from models import Pix2PixModel
from utils.checkpoint import save_checkpoint
from utils.seed import set_seed
from utils.visualize import save_comparison_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pix2Pix on paired edge/product images.")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--save_dir", default="./outputs/edge2shoes_100")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--direction", choices=["AtoB", "BtoA"], default="AtoB")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--vis_freq", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def choose_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return requested


def mean_dict(rows):
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / max(len(rows), 1) for key in keys}


def write_loss_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "G_loss", "D_loss", "GAN_loss", "L1_loss"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    save_dir = Path(args.save_dir)
    checkpoints_dir = save_dir / "checkpoints"
    samples_dir = save_dir / "samples"
    logs_dir = save_dir / "logs"
    for directory in (checkpoints_dir, samples_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset = PairedImageDataset(
        args.dataroot,
        split="train",
        img_size=args.img_size,
        direction=args.direction,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    fixed_batch = next(iter(loader))
    model = Pix2PixModel(
        lambda_l1=args.lambda_l1,
        lr=args.lr,
        beta1=args.beta1,
        device=device,
    )

    print(f"Training Pix2Pix on {len(dataset)} images | epochs={args.epochs} | device={device}")
    all_epoch_rows = []
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            model.set_input(batch)
            losses = model.optimize_parameters()
            epoch_losses.append(losses)
            progress.set_postfix({k: f"{v:.3f}" for k, v in losses.items()})

        averages = mean_dict(epoch_losses)
        row = {
            "epoch": epoch,
            "G_loss": averages["G_loss"],
            "D_loss": averages["D_loss"],
            "GAN_loss": averages["GAN_loss"],
            "L1_loss": averages["L1_loss"],
        }
        all_epoch_rows.append(row)
        write_loss_csv(logs_dir / "loss.csv", all_epoch_rows)

        save_checkpoint(model.netG, checkpoints_dir / "latest_G.pth", epoch=epoch, extra={"args": vars(args)})
        save_checkpoint(model.netD, checkpoints_dir / "latest_D.pth", epoch=epoch, extra={"args": vars(args)})
        if args.save_freq > 0 and epoch % args.save_freq == 0:
            save_checkpoint(model.netG, checkpoints_dir / f"epoch_{epoch:03d}_G.pth", epoch=epoch)
            save_checkpoint(model.netD, checkpoints_dir / f"epoch_{epoch:03d}_D.pth", epoch=epoch)

        if args.vis_freq > 0 and epoch % args.vis_freq == 0:
            with torch.no_grad():
                real_A = fixed_batch["input"].to(device)
                real_B = fixed_batch["target"].to(device)
                fake_B = model.netG(real_A)
            save_comparison_grid(
                real_A.cpu(),
                real_B.cpu(),
                fake_B.cpu(),
                samples_dir / f"epoch_{epoch:03d}.png",
                max_images=min(4, real_A.size(0)),
            )

        print(
            f"Epoch {epoch:03d}: G={row['G_loss']:.4f}, D={row['D_loss']:.4f}, "
            f"GAN={row['GAN_loss']:.4f}, L1={row['L1_loss']:.4f}"
        )

    summary = {
        "dataset_size": len(dataset),
        "epochs": args.epochs,
        "device": device,
        "latest_generator": str(checkpoints_dir / "latest_G.pth"),
        "latest_discriminator": str(checkpoints_dir / "latest_D.pth"),
        "loss_csv": str(logs_dir / "loss.csv"),
    }
    with open(logs_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Training complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
