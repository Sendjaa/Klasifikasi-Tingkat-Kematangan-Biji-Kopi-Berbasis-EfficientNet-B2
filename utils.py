# ============================================================
#  utils.py — Fungsi Pendukung (Metrics, Plot, Checkpoint, dll.)
# ============================================================

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)
import torch

import config


# ── AverageMeter ──────────────────────────────────────────
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ── Early Stopping ────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.stop      = False

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ── Logging ───────────────────────────────────────────────
def log_epoch(epoch, total, tr_loss, tr_acc, vl_loss, vl_acc, lr, elapsed):
    print(
        f"  Epoch [{epoch:>3}/{total}] "
        f"| Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} "
        f"| Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f} "
        f"| LR: {lr:.2e}  "
        f"| {elapsed:.1f}s"
    )


# ── Checkpoint ────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "val_loss":         val_loss,
        "val_acc":          val_acc,
        "model_name":       model.model_name,
        "num_classes":      model.num_classes,
    }, path)


# ── Plot Training Curves ──────────────────────────────────
def plot_training_curves(history: dict, save_path: str = None):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Kurva Pelatihan Model Deteksi Penyangraian Kopi",
                 fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss",
                 markersize=4)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",
                 markersize=4)
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc",
                 markersize=4)
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Acc",
                 markersize=4)
    axes[1].set_title("Akurasi"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(epochs, history["lr"], "g-o", markersize=4)
    axes[2].set_title("Learning Rate"); axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Kurva disimpan: {save_path}")
    plt.close()


# ── Confusion Matrix ──────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred,
                          class_names=None,
                          save_path=None,
                          normalize=True):
    if class_names is None:
        class_names = config.CLASSES

    # Ambil indeks kelas unik yang benar-benar ada di data aktual
    present_classes = np.unique(y_true)
    filtered_class_names = [class_names[i] for i in present_classes]

    # Hitung confusion matrix hanya untuk kelas yang hadir
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    
    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(8, 7))
    disp = sns.heatmap(
        cm_norm, annot=True,
        fmt=".2f" if normalize else "d",
        cmap="YlOrBr",
        xticklabels=filtered_class_names,
        yticklabels=filtered_class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Tingkat Penyangraian Kopi",
                 fontsize=13, pad=12)
    ax.set_xlabel("Prediksi", fontsize=11)
    ax.set_ylabel("Aktual",   fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Confusion Matrix disimpan: {save_path}")
    plt.close()
    return cm


# ── Classification Report ─────────────────────────────────
def print_classification_report(y_true, y_pred,
                                 class_names=None,
                                 save_path=None):
    if class_names is None:
        class_names = config.CLASSES

    # Ambil kelas unik yang benar-benar ada di data aktual
    present_classes = np.unique(y_true)
    filtered_class_names = [class_names[i] for i in present_classes]

    # Buat laporan klasifikasi yang aman tanpa crash akibat perbedaan dimensi
    report = classification_report(y_true, y_pred,
                                   labels=present_classes,
                                   target_names=filtered_class_names,
                                   digits=4)
    print("\n" + "="*60)
    print("  LAPORAN KLASIFIKASI")
    print("="*60)
    print(report)

    if save_path:
        with open(save_path, "w") as f:
            f.write(report)
        print(f"[Report] Disimpan: {save_path}")

    return report


# ── ROC-AUC Plot ──────────────────────────────────────────
def plot_roc_curves(y_true, y_prob,
                    class_names=None,
                    save_path=None):
    if class_names is None:
        class_names = config.CLASSES
        
    present_classes = np.unique(y_true)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Warna penanda grafik kurva ROC
    colors = ["#e74c3c", "#f39c12", "#8e44ad", "#2c3e50"]

    for i, cls_idx in enumerate(present_classes):
        cls_name = class_names[cls_idx]
        col = colors[i % len(colors)]
        
        # Buat label biner khusus untuk kelas aktif saat ini (one vs rest)
        y_bi = (y_true == cls_idx).astype(int)
        prob_bi = y_prob[:, cls_idx]
        
        fpr, tpr, _ = roc_curve(y_bi, prob_bi)
        auc_val = roc_auc_score(y_bi, prob_bi)
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{cls_name}  (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Kurva ROC — Tingkat Penyangraian Kopi")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] ROC Curve disimpan: {save_path}")
    plt.close()


# ── Grad-CAM (sederhana) ──────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model        = model
        self.target_layer = target_layer
        self._gradients   = None
        self._activations = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        layer = self.target_layer
        if layer is None:
            for m in self.model.backbone.modules():
                if isinstance(m, torch.nn.Conv2d):
                    layer = m
            self.target_layer = layer

        def fwd_hook(module, inp, out):
            self._activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hooks.append(layer.register_forward_hook(fwd_hook))
        self._hooks.append(layer.register_backward_hook(bwd_hook))

    def __call__(self, x: torch.Tensor,
                 class_idx: int = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(x)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        grads = self._gradients[0]          # (C, H, W)
        acts  = self._activations[0]        # (C, H, W)
        weights = grads.mean(dim=(1, 2))    # (C,)
        cam = (weights[:, None, None] * acts).sum(0).relu().numpy()

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


if __name__ == "__main__":
    y_t = np.random.randint(0, 3, 100)
    y_p = np.random.randint(0, 3, 100)
    cm = plot_confusion_matrix(y_t, y_p, class_names=['light', 'medium', 'dark'],
                               save_path=config.RESULTS_DIR / "demo_cm.png")
    print("Demo confusion matrix selesai.")