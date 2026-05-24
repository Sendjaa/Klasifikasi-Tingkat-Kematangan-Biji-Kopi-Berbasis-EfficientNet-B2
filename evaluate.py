# ============================================================
#  evaluate.py — Evaluasi Model pada Data Test
# ============================================================

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config
from dataset import CoffeeRoastDataset, get_transforms
from torch.utils.data import DataLoader
from model import load_model
from utils import (
    plot_confusion_matrix,
    print_classification_report,
    plot_roc_curves,
)


def evaluate_model(model_path: str = None):
    device = config.DEVICE
    if model_path is None:
        model_path = config.BEST_MODEL_PATH

    print(f"\n{'='*60}")
    print(f"  EVALUASI MODEL")
    print(f"  Checkpoint : {model_path}")
    print(f"  Device     : {device.upper()}")
    print(f"{'='*60}\n")

    # ── Model ──────────────────────────────────────────────
    model = load_model(str(model_path), device=device)
    model.eval()

    # ── DataLoader Test ────────────────────────────────────
    test_ds = CoffeeRoastDataset(
        config.TEST_DIR, split="test",
        transform=get_transforms("test")
    )
    if not test_ds.samples:
        print("[ERROR] Tidak ada data test. Tambahkan gambar ke data/test/<kelas>/")
        return

    test_loader = DataLoader(test_ds, batch_size=32,
                             shuffle=False, num_workers=4)

    # ── Inferensi ──────────────────────────────────────────
    all_preds   = []
    all_labels  = []
    all_probs   = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluasi", ncols=80):
            images = images.to(device)
            logits = model(images)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
            preds  = probs.argmax(axis=1)

            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_probs  = np.vstack(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Metrik ─────────────────────────────────────────────
    accuracy = (all_preds == all_labels).mean()
    print(f"\n  Akurasi Test : {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Memanggil fungsi cetak laporan dengan variabel yang benar
    print_classification_report(
        y_true=all_labels, 
        y_pred=all_preds, 
        class_names=config.CLASSES # <--- Masalahnya ada di baris ini
    )

    # ── Plotting Visualisasi ───────────────────────────────
    plot_confusion_matrix(
        all_labels, all_preds,
        class_names=config.CLASSES,
        save_path=config.RESULTS_DIR / "confusion_matrix.png",
    )

    plot_roc_curves(
        all_labels, all_probs,
        class_names=config.CLASSES,
        save_path=config.RESULTS_DIR / "roc_curves.png",
    )

    print(f"\n  Hasil evaluasi disimpan di: {config.RESULTS_DIR}/")
    return accuracy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Path ke file model .pth")
    args = parser.parse_args()
    evaluate_model(args.model)