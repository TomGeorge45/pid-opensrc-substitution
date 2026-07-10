"""Shared data-access helpers for the Stage 4 model bake-off.

Assumes Google Drive is already mounted at DRIVE_ROOT (Colab-specific — call
`from google.colab import drive; drive.mount('/content/drive')` before using this module).
"""

from pathlib import Path
from PIL import Image

DRIVE_ROOT = "/content/drive/MyDrive/pid_project/data"
KAGGLE_DIR = "kaggle_pid_symbols"
GUPTA_DIR = "gupta_pid"

ROOT = Path(DRIVE_ROOT)
kaggle_p = ROOT / KAGGLE_DIR
gupta_p = ROOT / GUPTA_DIR


def find_labeled_sample(min_boxes=3, max_boxes=8):
    """Returns (image_path, label_path, n_boxes) for one Kaggle tile with a ground-truth
    box count in range — informative for round-trip tests, unlike a blank tile."""
    for lbl in (kaggle_p / "labels").glob("*.txt"):
        n = len([l for l in lbl.read_text().splitlines() if l.strip()])
        if min_boxes <= n <= max_boxes:
            return kaggle_p / "images" / f"{lbl.stem}.jpg", lbl, n
    return None, None, 0


def load_sample():
    """Convenience wrapper: find a labeled sample and load the image."""
    img_path, label_path, gt_count = find_labeled_sample()
    img = Image.open(img_path).convert("RGB")
    return img, img_path, label_path, gt_count
