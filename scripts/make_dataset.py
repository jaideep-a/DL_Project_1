"""
make_dataset.py — Prepare or verify the chest X-ray dataset.

Place raw data at: data/raw/chest_xray/
Expected layout:
    data/raw/chest_xray/
        train/  NORMAL/  PNEUMONIA/
        val/    NORMAL/  PNEUMONIA/
        test/   NORMAL/  PNEUMONIA/


Usage:
    python scripts/make_dataset.py
"""

from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def verify_dataset(root: Path) -> None:
    """Check that chest_xray splits exist and report class counts."""
    chest = root / "data" / "raw" / "chest_xray"
    if not chest.exists():
        print(f"Dataset not found at {chest}")
        print("Download from: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia")
        return

    print(f"Dataset found: {chest}")
    for split in ("train", "val", "test"):
        split_path = chest / split
        if not split_path.exists():
            print(f"  Missing split: {split}")
            continue
        classes = [d.name for d in split_path.iterdir() if d.is_dir()]
        for cls in sorted(classes):
            count = len(list((split_path / cls).glob("*.*")))
            print(f"  {split}/{cls}: {count} images")


def main():
    """Entry point: verify dataset layout."""
    root = get_project_root()
    verify_dataset(root)


if __name__ == "__main__":
    main()
