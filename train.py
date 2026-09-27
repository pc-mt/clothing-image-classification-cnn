"""Train the clothing classifier and save the best validation checkpoint."""

from __future__ import annotations

import argparse
import copy
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from src import ClothingCNN, count_parameters

MEAN = [0.5065, 0.4740, 0.4509]
STD = [0.2158, 0.2072, 0.2007]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/clothing_clean_5classes_split_audit_copy"),
    )
    parser.add_argument("--output", type=Path, default=Path("models/best_clothing_cnn_final.pt"))
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_loaders(data_dir: Path, batch_size: int, workers: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(6),
            transforms.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.06),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    train_set = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_set = datasets.ImageFolder(data_dir / "val", transform=evaluation_transform)
    targets = [label for _, label in train_set.samples]
    counts = Counter(targets)
    weights = torch.DoubleTensor([1.0 / counts[label] for label in targets])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )
    return train_set, val_set, train_loader, val_loader


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    for split in ("train", "val", "test"):
        if not (args.data_dir / split).is_dir():
            raise FileNotFoundError(f"Missing dataset split: {args.data_dir / split}")

    train_set, _, train_loader, val_loader = build_loaders(
        args.data_dir, args.batch_size, args.workers
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClothingCNN(num_classes=len(train_set.classes)).to(device)
    assert count_parameters(model) <= 800_000
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_loss = float("inf")
    best_accuracy = 0.0
    best_state = None
    stale_epochs = 0

    print(f"Device: {device} | Parameters: {count_parameters(model):,}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_accuracy:.4f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": best_state,
                    "class_names": train_set.classes,
                    "mean": MEAN,
                    "std": STD,
                    "image_size": 224,
                    "epoch": epoch,
                    "val_loss": best_loss,
                    "val_accuracy": best_accuracy,
                },
                args.output,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Checkpoint: {args.output.resolve()}")


if __name__ == "__main__":
    main()
