from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PairedImageDataset(Dataset):
    """Dataset for Pix2Pix-style side-by-side paired images.

    For edges2shoes-style files, AtoB means left half is the sketch input and
    right half is the shoe photo target. BtoA reverses that mapping.
    """

    def __init__(
        self,
        dataroot,
        split="train",
        img_size=256,
        direction="AtoB",
        sample_size=None,
        seed=42,
        allow_unpaired=False,
    ):
        self.dataroot = Path(dataroot)
        self.split = split
        self.img_size = img_size
        self.direction = direction
        self.allow_unpaired = allow_unpaired
        if direction not in {"AtoB", "BtoA"}:
            raise ValueError("direction must be either AtoB or BtoA")

        data_dir = self._resolve_split_dir()
        self.paths = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)
        if sample_size is not None and sample_size > 0 and len(self.paths) > sample_size:
            rng = random.Random(seed)
            self.paths = sorted(rng.sample(self.paths, sample_size))
        if not self.paths:
            raise FileNotFoundError(f"No images found under {data_dir}")

    def _resolve_split_dir(self):
        candidates = []
        if self.split:
            candidates.append(self.dataroot / self.split)
        for name in ("test", "val", "train"):
            if self.split != name:
                candidates.append(self.dataroot / name)
        candidates.append(self.dataroot)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Dataset root does not exist: {self.dataroot}")

    def __len__(self):
        return len(self.paths)

    def _to_tensor(self, image):
        image = image.resize((self.img_size, self.img_size), Image.BICUBIC).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array.transpose(2, 0, 1))

    def _split_pair(self, image):
        width, height = image.size
        if width >= 2 and width >= height * 1.5:
            half = width // 2
            left = image.crop((0, 0, half, height))
            right = image.crop((half, 0, width, height))
            return left, right
        if self.allow_unpaired:
            return image, None
        raise ValueError(
            f"Expected a side-by-side paired image but got size {image.size}. "
            "Use allow_unpaired=True for input-only inference."
        )

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        left, right = self._split_pair(image)
        if right is None:
            source, target = left, None
        elif self.direction == "AtoB":
            source, target = left, right
        else:
            source, target = right, left

        item = {
            "input": self._to_tensor(source),
            "path": str(path),
            "name": path.stem,
        }
        if target is not None:
            item["target"] = self._to_tensor(target)
        return item
