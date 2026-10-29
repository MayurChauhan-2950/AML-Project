"""Central configuration: paths, limits, class names and shared text.

All paths are derived from this file's location, so the app works no matter
which directory ``uvicorn`` is started from.
"""
from __future__ import annotations

from pathlib import Path

BACKEND_DIR: Path = Path(__file__).resolve().parent.parent
REPO_ROOT: Path = BACKEND_DIR.parent

# --- Storage -----------------------------------------------------------
DATA_DIR: Path = BACKEND_DIR / "data"
DB_PATH: Path = DATA_DIR / "analyses.db"
UPLOAD_DIR: Path = DATA_DIR / "uploads"
HEATMAP_DIR: Path = DATA_DIR / "heatmaps"

# --- Models ------------------------------------------------------------
DEFAULT_MODEL = "resnet50"
MODEL_PATHS: dict[str, Path] = {
    "resnet50": REPO_ROOT / "resnet50_model.keras",
    "custom_cnn": REPO_ROOT / "custom_cnn_model.keras",
}
MODEL_VERSIONS: dict[str, str] = {
    "resnet50": "resnet50-v1",
    "custom_cnn": "custom_cnn-v1",
}
# Last convolutional layer used for Grad-CAM (found by reading config.json of
# each .keras file, see README). The Grad-CAM code falls back to "the last
# layer with a 4-D output" if a name is not found.
GRADCAM_LAYERS: dict[str, str] = {
    "resnet50": "conv5_block3_out",
    "custom_cnn": "conv2d_2",
}
METRICS_CSV: Path = REPO_ROOT / "model_comparison_results.csv"

# --- Classes (alphabetical, as produced by image_dataset_from_directory) --
CLASSES: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABELS: dict[str, str] = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}
IMG_SIZE: tuple[int, int] = (224, 224)

# --- Upload limits -----------------------------------------------------
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})
# Pillow format name -> (stored extension, media type)
ALLOWED_FORMATS: dict[str, tuple[str, str]] = {
    "PNG": (".png", "image/png"),
    "JPEG": (".jpg", "image/jpeg"),
}
MAX_IMAGE_PIXELS = 50_000_000  # reject decompression bombs

# --- Misc --------------------------------------------------------------
CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
DISCLAIMER = (
    "AI-generated prediction for educational and research use only. It is not a "
    "medical diagnosis and must not replace review by a qualified clinician."
)
