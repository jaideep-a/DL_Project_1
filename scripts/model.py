"""
model.py — Train model and run predictions.

This script is the command-line interface for training the pneumonia
detection model and running offline predictions on images.

For live predictions the web application (main.py -> app/) calls the
external model API configured via MODEL_API_URL in .env.


Usage:
    python scripts/model.py --train
    python scripts/model.py --predict path/to/image.jpg
"""

import argparse
from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def train(data_dir: Path, model_dir: Path) -> None:
    """
    Train the pneumonia detection model.

    Args:
        data_dir: Path to processed data directory.
        model_dir: Path to save trained model artifacts.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        print(f"Data not found at {data_dir}. Run build_features.py first.")
        return
    print(f"Training model using data from {data_dir}")
    print(f"Model artifacts will be saved to {model_dir}")
    print("Implement training logic here (scikit-learn, PyTorch, etc.)")


def predict(image_path: Path) -> None:
    """
    Run offline prediction on a single image.

    Args:
        image_path: Path to the chest X-ray image.
    """
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return
    print(f"Predicting: {image_path}")
    print("For live predictions, use the web app at http://localhost:5000")


def main():
    """Entry point: parse args and run train or predict."""
    parser = argparse.ArgumentParser(
        description="Train model or run offline prediction"
    )
    parser.add_argument("--train", action="store_true", help="Run model training")
    parser.add_argument("--predict", type=str, metavar="IMAGE", help="Path to image for prediction")
    args = parser.parse_args()

    root = get_project_root()

    if args.train:
        train(
            data_dir=root / "data" / "processed",
            model_dir=root / "models"
        )
    elif args.predict:
        predict(Path(args.predict))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
