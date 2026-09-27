"""Evaluate a saved checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import ClothingCNN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/clothing_clean_5classes_split_audit_copy"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_clothing_cnn_final.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    classes = checkpoint.get("class_names", ["long-sleeved", "pants", "shorts", "socks", "t-shirt"])
    transform = transforms.Compose(
        [
            transforms.Resize((checkpoint.get("image_size", 224),) * 2),
            transforms.ToTensor(),
            transforms.Normalize(
                checkpoint.get("mean", [0.5065, 0.4740, 0.4509]),
                checkpoint.get("std", [0.2158, 0.2072, 0.2007]),
            ),
        ]
    )
    dataset = datasets.ImageFolder(args.data_dir / "test", transform=transform)
    if dataset.classes != classes:
        raise ValueError(f"Checkpoint classes {classes} do not match dataset classes {dataset.classes}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = ClothingCNN(len(classes))
    model.load_state_dict(state_dict)
    model.eval()
    expected, predicted = [], []
    with torch.inference_mode():
        for images, labels in loader:
            expected.extend(labels.tolist())
            predicted.extend(model(images).argmax(1).tolist())
    print(f"Accuracy: {accuracy_score(expected, predicted):.4f}")
    print(classification_report(expected, predicted, target_names=classes, digits=4))
    print("Confusion matrix:\n", confusion_matrix(expected, predicted))


if __name__ == "__main__":
    main()
