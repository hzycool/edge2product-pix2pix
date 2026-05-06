import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_losses(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items() if k})
    return rows


def plot_loss(csv_path, save_path):
    rows = read_losses(csv_path)
    if not rows:
        raise ValueError(f"No loss rows found in {csv_path}")
    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(8, 5))
    for key in ("G_loss", "D_loss", "GAN_loss", "L1_loss"):
        if key in rows[0]:
            plt.plot(epochs, [row[key] for row in rows], marker="o", label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pix2Pix Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    return save_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--save_path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = plot_loss(args.csv_path, args.save_path)
    print(f"Saved loss curve to {path}")
