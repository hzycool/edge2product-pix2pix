from pathlib import Path

import torch


def save_checkpoint(model, path, epoch=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict()}
    if epoch is not None:
        payload["epoch"] = epoch
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_generator(generator, checkpoint_path, device="cpu", strict=True):
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("state_dict", payload)
    generator.load_state_dict(state_dict, strict=strict)
    return generator
