from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def tensor_to_pil(tensor):
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = tensor.detach().cpu().float().clamp(-1, 1).numpy()
    array = ((array + 1.0) * 127.5).round().astype(np.uint8)
    array = array.transpose(1, 2, 0)
    return Image.fromarray(array)


def save_comparison_grid(inputs, targets, generated, save_path, max_images=4, labels=None):
    labels = labels or ["input", "target", "generated"]
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(max_images, inputs.size(0))
    cols = 3 if targets is not None else 2
    cell_w, cell_h = 256, 286
    grid = Image.new("RGB", (cols * cell_w, n * cell_h), "white")
    draw = ImageDraw.Draw(grid)

    for row in range(n):
        images = [tensor_to_pil(inputs[row])]
        if targets is not None:
            images.append(tensor_to_pil(targets[row]))
        images.append(tensor_to_pil(generated[row]))
        for col, image in enumerate(images):
            image = image.resize((256, 256), Image.BICUBIC)
            x, y = col * cell_w, row * cell_h
            grid.paste(image, (x, y + 24))
            draw.text((x + 8, y + 6), labels[col], fill=(0, 0, 0))

    grid.save(save_path)
    return save_path
