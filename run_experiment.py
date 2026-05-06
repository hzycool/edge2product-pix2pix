import json
import shutil
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Edge2Product Pix2Pix experiment pipeline.")
    parser.add_argument(
        "--no_cpu_downgrade",
        action="store_true",
        help="Run the configured sample_size/epochs even on CPU. This can take much longer.",
    )
    return parser.parse_args()


def run(cmd):
    print("Running:", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], cwd=ROOT, check=True)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def has_images(path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    path = Path(path)
    return path.exists() and any(p.suffix.lower() in exts for p in path.rglob("*"))


def write_log(log_path, content):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(content)


def copy_if_exists(src, dst):
    src, dst = Path(src), Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def update_report(metrics_path, log_note):
    report_path = ROOT / "report" / "report.tex"
    if not report_path.exists():
        return
    text = report_path.read_text(encoding="utf-8")
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8")) if Path(metrics_path).exists() else None
    if metrics:
        table = (
            "% EDGE2PRODUCT_METRICS_TABLE_START\n"
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\caption{小样本实验的定量评价结果}\n"
            "\\begin{tabular}{lccc}\n"
            "\\toprule\n"
            "样本对数 & Mean L1 $\\downarrow$ & PSNR $\\uparrow$ & SSIM $\\uparrow$ \\\\\n"
            "\\midrule\n"
            f"{metrics['num_pairs']} & {metrics['mean_l1']:.4f} & {metrics['psnr']:.2f} & {metrics['ssim']:.4f} \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\label{tab:metrics}\n"
            "\\end{table}\n"
            "% EDGE2PRODUCT_METRICS_TABLE_END"
        )
    else:
        table = (
            "% EDGE2PRODUCT_METRICS_TABLE_START\n"
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\caption{定量评价结果待实验运行后自动填入}\n"
            "\\begin{tabular}{lccc}\n"
            "\\toprule\n"
            "样本对数 & Mean L1 $\\downarrow$ & PSNR $\\uparrow$ & SSIM $\\uparrow$ \\\\\n"
            "\\midrule\n"
            "待填入 & 待填入 & 待填入 & 待填入 \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\label{tab:metrics}\n"
            "\\end{table}\n"
            "% EDGE2PRODUCT_METRICS_TABLE_END"
        )
    text = replace_block(text, "% EDGE2PRODUCT_METRICS_TABLE_START", "% EDGE2PRODUCT_METRICS_TABLE_END", table)
    note = (
        "% EDGE2PRODUCT_EXPERIMENT_NOTE_START\n"
        f"\\noindent\\textbf{{实验记录：}}{log_note}\n"
        "% EDGE2PRODUCT_EXPERIMENT_NOTE_END"
    )
    text = replace_block(text, "% EDGE2PRODUCT_EXPERIMENT_NOTE_START", "% EDGE2PRODUCT_EXPERIMENT_NOTE_END", note)
    report_path.write_text(text, encoding="utf-8")


def replace_block(text, start, end, replacement):
    start_idx = text.find(start)
    end_idx = text.find(end)
    if start_idx == -1 or end_idx == -1:
        return text
    end_idx += len(end)
    return text[:start_idx] + replacement + text[end_idx:]


def main():
    args = parse_args()
    config = load_config(ROOT / "configs" / "pix2pix_edges2shoes.yaml")
    data_cfg = config["dataset"]
    train_cfg = config["train"]
    exp_cfg = config["experiment"]
    source_root = ROOT / data_cfg["root"]
    subset_root = ROOT / data_cfg["subset_root"]
    output_root = ROOT / exp_cfg["output_root"]
    log_path = ROOT / "docs" / "experiment_log.md"

    if not has_images(source_root):
        message = (
            "# Experiment Log\n\n"
            f"- Time: {datetime.now().isoformat(timespec='seconds')}\n"
            "- Status: dataset missing.\n"
            "- Please run: `bash scripts/download_edges2shoes.sh`\n\n"
            "当前环境未发现 edges2shoes 数据集，因此未运行训练、推理和评估。"
        )
        write_log(log_path, message)
        print("Dataset not found. Please run: bash scripts/download_edges2shoes.sh")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_size = int(train_cfg["sample_size"])
    epochs = int(train_cfg["epochs"])
    resource_note = ""
    if device == "cpu" and not args.no_cpu_downgrade:
        sample_size = min(sample_size, 10)
        epochs = 1
        resource_note = "受限于本机 CPU 计算环境，本次运行采用较小训练规模；后续可在 GPU 环境下扩大样本量和训练轮数。"
    elif device == "cpu":
        resource_note = "本次实验在 CPU 环境下按配置规模运行，因此训练耗时较长；实验结果可用于观察默认设置下的完整流程。"

    run([
        sys.executable,
        "make_subset.py",
        "--dataroot",
        source_root,
        "--sample_size",
        sample_size,
        "--output_root",
        subset_root,
    ])
    run([
        sys.executable,
        "train.py",
        "--dataroot",
        subset_root,
        "--save_dir",
        output_root,
        "--epochs",
        epochs,
        "--batch_size",
        train_cfg["batch_size"],
        "--lr",
        train_cfg["lr"],
        "--lambda_l1",
        train_cfg["lambda_l1"],
        "--img_size",
        data_cfg["img_size"],
        "--direction",
        data_cfg["direction"],
        "--device",
        device,
        "--sample_size",
        sample_size,
    ])
    run([
        sys.executable,
        "infer.py",
        "--dataroot",
        subset_root,
        "--checkpoint",
        output_root / "checkpoints" / "latest_G.pth",
        "--save_dir",
        output_root / "inference",
        "--img_size",
        data_cfg["img_size"],
        "--direction",
        data_cfg["direction"],
        "--device",
        device,
        "--num_images",
        exp_cfg["num_inference_images"],
    ])
    run([
        sys.executable,
        "evaluate.py",
        "--generated_dir",
        output_root / "inference" / "generated",
        "--target_dir",
        output_root / "inference" / "target",
        "--save_path",
        output_root / "metrics" / "metrics.json",
    ])
    run([
        sys.executable,
        "utils/plot_loss.py",
        "--csv_path",
        output_root / "logs" / "loss.csv",
        "--save_path",
        output_root / "curves" / "loss_curve.png",
    ])

    copy_if_exists(output_root / "curves" / "loss_curve.png", ROOT / "report" / "figures" / "loss_curve.png")
    copy_if_exists(output_root / "inference" / "inference_grid.png", ROOT / "report" / "figures" / "inference_grid.png")
    copy_if_exists(ROOT / "assets" / "ui_screenshot_placeholder.png", ROOT / "report" / "figures" / "ui_screenshot_placeholder.png")

    metrics_path = output_root / "metrics" / "metrics.json"
    log_note = (
        f"本次实验在 {device} 环境下运行，训练样本数为 {sample_size}，训练轮数为 {epochs}。"
        + (resource_note if resource_note else "实验结果见表 \\ref{tab:metrics}。")
    )
    update_report(metrics_path, log_note)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    log = (
        "# Experiment Log\n\n"
        f"- Time: {datetime.now().isoformat(timespec='seconds')}\n"
        f"- Device: {device}\n"
        f"- Sample size: {sample_size}\n"
        f"- Epochs: {epochs}\n"
        f"- Mean L1: {metrics['mean_l1']:.6f}\n"
        f"- PSNR: {metrics['psnr']:.4f}\n"
        f"- SSIM: {metrics['ssim']:.4f}\n\n"
        f"{resource_note}\n"
    )
    write_log(log_path, log)
    print(f"Experiment complete. Log written to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
