# ============================================================
#  config.py — Konfigurasi Proyek Deteksi Penyangraian Kopi
# ============================================================

import os
from pathlib import Path

# --- Direktori ---
BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data"
TRAIN_DIR      = DATA_DIR / "train"
VAL_DIR        = DATA_DIR / "val"
TEST_DIR       = DATA_DIR / "test"
MODEL_DIR      = BASE_DIR / "models"
RESULTS_DIR    = BASE_DIR / "results"

# --- Kelas Tingkat Penyangraian ---
CLASSES = ["light", "medium","medium_dark", "dark"]
CLASS_NAMES_ID = {
    "light":       "Light Roast (Sangrai Ringan)",
    "medium":      "Medium Roast (Sangrai Sedang)",
    "medium_dark": "Medium-Dark Roast (Sangrai Sedang-Gelap)",
    "dark":        "Dark Roast (Sangrai Gelap)",
}
NUM_CLASSES = len(CLASSES)

# --- Deskripsi Kelas ---
CLASS_DESCRIPTIONS = {
    "light": {
        "warna": "Cokelat muda / Cinnamon",
        "suhu":  "196–205°C",
        "rasa":  "Asam tinggi, fruity, floral",
        "aroma": "Buah segar, bunga",
        "kafein": "Tinggi",
    },
    "medium": {
        "warna": "Cokelat sedang",
        "suhu":  "210–220°C",
        "rasa":  "Seimbang, sedikit manis, caramel",
        "aroma": "Karamel, kacang",
        "kafein": "Sedang",
    },
    "medium_dark": {
        "warna": "Cokelat tua / kecokelatan",
        "suhu":  "225–230°C",
        "rasa":  "Pahit ringan, cokelat, sedikit berminyak",
        "aroma": "Dark chocolate, rempah",
        "kafein": "Sedang-rendah",
    },
    "dark": {
        "warna": "Hitam / sangat gelap",
        "suhu":  "240–250°C",
        "rasa":  "Pahit kuat, smoky, karbonat",
        "aroma": "Smoky, terbakar, pekat",
        "kafein": "Rendah",
    },
}

# --- Hyperparameter ---
BATCH_SIZE      = 32
NUM_EPOCHS      = 30
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
DROPOUT_RATE    = 0.4
PATIENCE        = 7        # Early stopping patience
MIN_DELTA       = 1e-4     # Minimum improvement threshold

# --- Arsitektur Model ---
MODEL_NAME      = "efficientnet_b2"   # efficientnet_b0 / resnet50 / mobilenetv3_large_100
PRETRAINED      = True
FREEZE_LAYERS   = 0        # Jumlah layer yang dibekukan (0 = fine-tune semua)

# --- Augmentasi ---
IMAGE_SIZE      = 224
MEAN            = [0.485, 0.456, 0.406]   # ImageNet mean
STD             = [0.229, 0.224, 0.225]   # ImageNet std

# --- Device ---
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Simpan Model ---
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
LAST_MODEL_PATH = MODEL_DIR / "last_model.pth"

# Pastikan folder ada
for d in [MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    