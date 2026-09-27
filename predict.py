"""Predict the clothing class of one image."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src import ClothingCNN

DEFAULT_CLASSES = ["long-sleeved", "pants", "shorts", "socks", "t-shirt"]
DEFAULT_MEAN = [0.5065, 0.4740, 0.4509]
DEFAULT_STD = [0.2158, 0.2072, 0.2007]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_clothing_cnn_final.pt"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    classes = checkpoint.get("class_names", DEFAULT_CLASSES)
    mean = checkpoint.get("mean", DEFAULT_MEAN)
    std = checkpoint.get("std", DEFAULT_STD)
    image_size = checkpoint.get("image_size", 224)
    model = ClothingCNN(num_classes=len(classes))
    model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(image), dim=1)[0]
    values, indices = torch.topk(probabilities, k=min(3, len(classes)))
    for value, index in zip(values.tolist(), indices.tolist()):
        print(f"{classes[index]:<14} {value:.2%}")


if __name__ == "__main__":
    main()
