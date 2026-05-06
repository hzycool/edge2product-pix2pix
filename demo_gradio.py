import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import gradio as gr

from models import GeneratorUNet
from utils.checkpoint import load_generator
from utils.visualize import tensor_to_pil


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio demo for Edge2Product Pix2Pix.")
    parser.add_argument("--checkpoint", default="./outputs/edge2shoes_100/checkpoints/latest_G.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--save_dir", default="./outputs/gradio_results")
    parser.add_argument("--server_name", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    return parser.parse_args()


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def pil_to_tensor(image, img_size):
    image = image.convert("RGB").resize((img_size, img_size), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def build_predictor(args):
    device = choose_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    generator = GeneratorUNet().to(device)
    status = f"Checkpoint loaded: {checkpoint}"
    if checkpoint.exists():
        load_generator(generator, checkpoint, device=device)
        generator.eval()
    else:
        status = f"Checkpoint not found: {checkpoint}. Please train a model first."
        generator = None

    def predict(image):
        if image is None:
            return None, "Please upload an edge sketch first."
        if generator is None:
            return None, status
        with torch.no_grad():
            tensor = pil_to_tensor(image, args.img_size).to(device)
            output = generator(tensor)
        result = tensor_to_pil(output.cpu())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = save_dir / f"edge2product_{timestamp}.png"
        result.save(save_path)
        return result, f"Saved result to {save_path}"

    return predict, status


def main():
    args = parse_args()
    predict, status = build_predictor(args)
    examples = []
    for candidate in ("assets/example_sketch.png", "assets/teaser_placeholder.png"):
        if Path(candidate).exists():
            examples.append(candidate)

    with gr.Blocks(title="Edge2Product Pix2Pix") as demo:
        gr.Markdown("# Edge2Product: Sketch-to-Product Generation with Pix2Pix")
        gr.Markdown(
            "This demo uses a Pix2Pix conditional GAN to translate edge sketches into "
            "product-style shoe images."
        )
        gr.Markdown(f"**Checkpoint status:** {status}")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload edge sketch", type="pil", image_mode="RGB")
                generate_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Generated product image", type="pil")
                result_path = gr.Textbox(label="Result save path", interactive=False)
        if examples:
            gr.Examples(examples=examples, inputs=input_image)
        generate_btn.click(fn=predict, inputs=input_image, outputs=[output_image, result_path])
        gr.Markdown(
            "This project is a course-level Pix2Pix GAN demo and does not aim for "
            "production-level image quality."
        )

    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
