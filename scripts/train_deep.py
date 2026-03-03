"""
Deep Learning Model — ResNet18 Fine-Tuning
==========================================
Fine-tunes a pre-trained ResNet18 on chest X-ray images for binary
classification: NORMAL vs PNEUMONIA.

Uses:
- Transfer learning from ImageNet weights
- Weighted cross-entropy loss to handle class imbalance
- Data augmentation (flip, rotation) for better generalisation


Usage:
    python scripts/train_deep.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from pathlib import Path
from PIL import Image
import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "chest_xray"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

CLASSES = ["NORMAL", "PNEUMONIA"]
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ChestXRayDataset(Dataset):
    """PyTorch Dataset for chest X-ray images."""

    def __init__(self, split_dir: Path, transform=None):
        """
        Args:
            split_dir: Path to train/val/test directory.
            transform: Torchvision transforms to apply.
        """
        self.transform = transform
        self.samples = []  # (path, label_idx)

        valid_exts = {".jpg", ".jpeg", ".png"}
        for label_idx, cls in enumerate(CLASSES):
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.glob("*.*"):
                if img_path.suffix.lower() in valid_exts:
                    self.samples.append((img_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_class_weights(dataset: ChestXRayDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights to handle imbalance."""
    counts = [0] * len(CLASSES)
    for _, label in dataset.samples:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (len(CLASSES) * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float)


def build_model(num_classes: int = 2) -> nn.Module:
    """Load ResNet18 with pretrained ImageNet weights, replace head."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    """Compute accuracy on a dataloader."""
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total if total > 0 else 0

    # Per-class accuracy
    class_correct = [0] * len(CLASSES)
    class_total = [0] * len(CLASSES)
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    per_class = {
        CLASSES[i]: round(class_correct[i] / class_total[i], 4)
        for i in range(len(CLASSES))
        if class_total[i] > 0
    }

    return {"accuracy": round(accuracy, 4), "per_class": per_class}


def main():
    print("=" * 55)
    print(f"Deep Learning — ResNet18 Fine-Tuning  [{DEVICE}]")
    print("=" * 55)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Datasets
    print("\nLoading datasets...")
    train_dataset = ChestXRayDataset(DATA_DIR / "train", transform=train_transform)
    test_dataset = ChestXRayDataset(DATA_DIR / "test", transform=test_transform)
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Test : {len(test_dataset)} images")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    print("\nBuilding ResNet18 model...")
    model = build_model(num_classes=len(CLASSES)).to(DEVICE)

    # Weighted loss for class imbalance
    class_weights = get_class_weights(train_dataset).to(DEVICE)
    print(f"  Class weights: NORMAL={class_weights[0]:.2f}, PNEUMONIA={class_weights[1]:.2f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS}  "
                      f"Step {batch_idx+1}/{len(train_loader)}  "
                      f"Loss: {running_loss/(batch_idx+1):.4f}  "
                      f"Acc: {correct/total*100:.1f}%")

        train_acc = correct / total
        print(f"  → Epoch {epoch+1} train accuracy: {train_acc*100:.1f}%")

    # Final evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader)
    print(f"  Test accuracy : {test_metrics['accuracy']*100:.1f}%")
    print(f"  Per class     : {test_metrics['per_class']}")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "deep_resnet18.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "image_size": IMAGE_SIZE,
            "metrics": test_metrics,
        },
        model_path,
    )
    print(f"\nModel saved to {model_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
