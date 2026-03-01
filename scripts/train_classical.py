"""
Classical ML Model — HOG Features + SVM Classifier
===================================================
Extracts Histogram of Oriented Gradients (HOG) features from chest
X-ray images and trains a Support Vector Machine (SVM) classifier.

AI Attribution: Script written with assistance from Claude AI (Anthropic), Feb 2026.

Usage:
    python scripts/train_classical.py
"""

import os
import joblib
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "chest_xray"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

IMAGE_SIZE = (64, 64)       # Resize target
MAX_PER_CLASS = 1000        # Subsample cap to keep training fast
CLASSES = ["NORMAL", "PNEUMONIA"]


def load_images_and_labels(split_dir: Path, max_per_class: int = None) -> tuple:
    """
    Load images from a split directory, extract HOG features.

    Args:
        split_dir: Path to train/val/test directory.
        max_per_class: Maximum images to load per class (for speed).

    Returns:
        Tuple of (features array, labels list).
    """
    features, labels = [], []

    for cls in CLASSES:
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue

        image_paths = list(cls_dir.glob("*.*"))
        if max_per_class:
            image_paths = image_paths[:max_per_class]

        print(f"  Loading {cls}: {len(image_paths)} images...")

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("L")        # Grayscale
                img = img.resize(IMAGE_SIZE)
                img_array = np.array(img, dtype=np.float32) / 255.0

                # Extract HOG features
                feat = hog(
                    img_array,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm="L2-Hys",
                )
                features.append(feat)
                labels.append(cls)

            except Exception:
                continue

    return np.array(features), labels


def main():
    print("=" * 55)
    print("Classical ML — HOG Features + SVM")
    print("=" * 55)

    # Load training data
    print(f"\nLoading training data (up to {MAX_PER_CLASS}/class)...")
    X_train, y_train = load_images_and_labels(
        DATA_DIR / "train", max_per_class=MAX_PER_CLASS
    )
    print(f"  Training samples: {len(X_train)}, feature dim: {X_train.shape[1]}")

    # Load test data
    print("\nLoading test data...")
    X_test, y_test = load_images_and_labels(DATA_DIR / "test")
    print(f"  Test samples: {len(X_test)}")

    # Feature scaling
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    print("\nTraining SVM (this may take ~1-2 minutes)...")
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True)
    svm.fit(X_train_scaled, y_train)

    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASSES)

    print(f"  Test accuracy: {accuracy * 100:.1f}%")
    print(f"\nClassification Report:\n{report}")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "classical_svm.pkl"
    joblib.dump(
        {
            "model": svm,
            "scaler": scaler,
            "classes": CLASSES,
            "image_size": IMAGE_SIZE,
            "metrics": {"accuracy": round(accuracy, 4)},
        },
        model_path,
    )
    print(f"Model saved to {model_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
