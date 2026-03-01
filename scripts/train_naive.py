"""
Naive Baseline Model — Most Frequent Class Classifier
=====================================================
Predicts the most common class in the training set for every input.
Used as a lower-bound benchmark for more sophisticated models.

AI Attribution: Script written with assistance from Claude AI (Anthropic), Feb 2026.

Usage:
    python scripts/train_naive.py
"""

import os
import joblib
from pathlib import Path
from collections import Counter


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "chest_xray"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def count_classes(split_dir: Path) -> Counter:
    """Count images per class in a split directory."""
    counts = Counter()
    for cls_dir in split_dir.iterdir():
        if cls_dir.is_dir():
            n = len([f for f in cls_dir.iterdir() if f.is_file()])
            counts[cls_dir.name] = n
    return counts


class MostFrequentClassifier:
    """Always predicts the most frequent class seen in training."""

    def __init__(self):
        self.most_frequent_class = None
        self.class_counts = {}

    def fit(self, class_counts: dict) -> None:
        """Set the most frequent class from training label counts."""
        self.class_counts = class_counts
        self.most_frequent_class = max(class_counts, key=class_counts.get)

    def predict(self, n_samples: int = 1) -> list:
        """Return the most frequent class for every sample."""
        return [self.most_frequent_class] * n_samples

    def evaluate(self, test_dir: Path) -> dict:
        """Compute accuracy on test set (always predicts majority class)."""
        test_counts = count_classes(test_dir)
        total = sum(test_counts.values())
        correct = test_counts.get(self.most_frequent_class, 0)
        accuracy = correct / total if total > 0 else 0
        return {
            "accuracy": round(accuracy, 4),
            "total_samples": total,
            "correct": correct,
            "predicted_class": self.most_frequent_class,
            "class_distribution": dict(test_counts),
        }


def main():
    print("=" * 55)
    print("Naive Baseline — Most Frequent Class Classifier")
    print("=" * 55)

    train_dir = DATA_DIR / "train"
    test_dir = DATA_DIR / "test"

    # Count training classes
    print("\nCounting training classes...")
    train_counts = count_classes(train_dir)
    for cls, n in train_counts.items():
        print(f"  {cls}: {n} images")

    # Train (trivially: just record the majority class)
    model = MostFrequentClassifier()
    model.fit(dict(train_counts))
    print(f"\nMost frequent class: {model.most_frequent_class}")
    print("Naive model always predicts this class.")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = model.evaluate(test_dir)
    print(f"  Test accuracy : {results['accuracy'] * 100:.1f}%")
    print(f"  Correct       : {results['correct']} / {results['total_samples']}")
    print(f"  Distribution  : {results['class_distribution']}")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "naive_baseline.pkl"
    joblib.dump({"model": model, "metrics": results}, model_path)
    print(f"\nModel saved to {model_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
