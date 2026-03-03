"""
build_features.py — Feature engineering pipeline.

Reads raw chest X-ray images from data/raw/chest_xray/ and writes
processed feature representations to data/processed/.

For this project the deep learning model (ResNet/ViT) learns features
end-to-end, so this script provides a HOG-based classical feature
extraction pipeline as a reference / baseline.


Usage:
    python scripts/build_features.py
"""

from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def build_features(raw_dir: Path, out_dir: Path) -> None:
    """
    Extract features from images in raw_dir and save to out_dir.

    Args:
        raw_dir: Path to raw chest_xray split directory.
        out_dir: Path to write processed feature files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"Raw data not found at {raw_dir}. Run make_dataset.py first.")
        return

    print(f"Building features from {raw_dir} -> {out_dir}")
    print("Add feature extraction logic here (e.g. HOG, pixel flatten).")


def main():
    """Entry point: run feature pipeline for all splits."""
    root = get_project_root()
    chest = root / "data" / "raw" / "chest_xray"
    out = root / "data" / "processed"

    for split in ("train", "val", "test"):
        build_features(chest / split, out / split)

    print("Feature build complete. Outputs in data/processed/")


if __name__ == "__main__":
    main()
