# ============================================================
#  train.py — Pipeline Pelatihan Model
# ============================================================

import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR
)
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import build_model, LabelSmoothingCrossEntropy, FocalLoss
from utils import (
    AverageMeter, EarlyStopping,
    plot_training_curves, save_checkpoint, log_epoch
)


# ══════════════════════════════════════════════════════════
#  Satu epoch train
# ══════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    losses  = AverageMeter()
    correct = 0
    total   = 0

    pbar = tqdm(loader, desc="  [Train]", leave=False, ncols=90)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(enabled=(device == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        preds    = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        losses.update(loss.item(), labels.size(0))

        pbar.set_postfix({"loss": f"{losses.avg:.4f}",
                          "acc":  f"{correct/total:.4f}"})

    return losses.avg, correct / total


# ══════════════════════════════════════════════════════════
#  Evaluasi (val / test)
# ══════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses  = AverageMeter()
    correct = 0
    total   = 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  [Eval] ", leave=False, ncols=90):
        images, labels = images.to(device), labels.to(device)
        with autocast(enabled=(device == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        preds    = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        losses.update(loss.item(), labels.size(0))

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return losses.avg, correct / total, np.array(all_preds), np.array(all_labels)


# ══════════════════════════════════════════════════════════
#  Loop Pelatihan Utama
# ══════════════════════════════════════════════════════════
def train(
    model_name:   str   = config.MODEL_NAME,
    num_epochs:   int   = config.NUM_EPOCHS,
    lr:           float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    loss_fn:      str   = "label_smoothing",   # 'ce' | 'label_smoothing' | 'focal'
    use_amp:      bool  = True,
    resume:       str   = None,
):
    device = config.DEVICE
    print(f"\n{'='*60}")
    print(f"  COFFEE ROAST DETECTION — Pelatihan")
    print(f"  Device  : {device.upper()}")
    print(f"  Backbone: {model_name}")
    print(f"  Epochs  : {num_epochs}  |  LR: {lr}")
    print(f"{'='*60}\n")

    # ── DataLoader ─────────────────────────────────────────
    loaders = get_dataloaders()
    if not loaders["train"].dataset.samples:
        print("[ERROR] Dataset kosong. Letakkan gambar di data/train/<kelas>/")
        return

    # ── Model ──────────────────────────────────────────────
    model = build_model(device=device, model_name=model_name)

    # Resume dari checkpoint
    start_epoch = 0
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[Resume] Melanjutkan dari epoch {start_epoch}")

    # ── Loss ───────────────────────────────────────────────
    class_weights = loaders["train"].dataset.get_class_weights().to(device)
    if loss_fn == "label_smoothing":
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    elif loss_fn == "focal":
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer ──────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ── Scheduler: warmup + cosine ─────────────────────────
    warmup_epochs = max(1, num_epochs // 10)
    warmup = LinearLR(optimizer,
                      start_factor=0.01, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer,
                               T_max=num_epochs - warmup_epochs,
                               eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                              milestones=[warmup_epochs])

    # ── Mixed Precision ────────────────────────────────────
    scaler = GradScaler(enabled=(use_amp and device == "cuda"))

    # ── Early Stopping ─────────────────────────────────────
    early_stop = EarlyStopping(patience=config.PATIENCE,
                               min_delta=config.MIN_DELTA)

    # ── History ────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr":         [],
    }
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # ══════════════════════════════════════════════════════
    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        tr_loss, tr_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device
        )
        # Val
        vl_loss, vl_acc, _, _ = evaluate(
            model, loaders["val"], criterion, device
        )

        scheduler.step()

        # Catat history
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        log_epoch(epoch + 1, num_epochs,
                  tr_loss, tr_acc, vl_loss, vl_acc,
                  current_lr, elapsed)

        # Simpan model terbaik
        if vl_acc > best_val_acc + config.MIN_DELTA:
            best_val_acc = vl_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint(model, optimizer, epoch,
                            vl_loss, vl_acc,
                            config.BEST_MODEL_PATH)
            print(f"    ✓ Model terbaik disimpan (val_acc={vl_acc:.4f})")

        # Simpan checkpoint terakhir
        save_checkpoint(model, optimizer, epoch,
                        vl_loss, vl_acc,
                        config.LAST_MODEL_PATH)

        # Early stopping
        early_stop(vl_loss)
        if early_stop.stop:
            print(f"\n[Early Stop] Pelatihan dihentikan di epoch {epoch+1}")
            break
    # ══════════════════════════════════════════════════════

    # Muat bobot terbaik kembali
    model.load_state_dict(best_model_wts)
    print(f"\n{'='*60}")
    print(f"  Pelatihan Selesai! Val Acc Terbaik: {best_val_acc:.4f}")
    print(f"{'='*60}")

    # Simpan history
    hist_path = config.RESULTS_DIR / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    # Plot kurva
    plot_training_curves(history,
                         save_path=config.RESULTS_DIR / "training_curves.png")

    return model, history


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Latih model deteksi tingkat penyangraian kopi")
    parser.add_argument("--model",    default=config.MODEL_NAME)
    parser.add_argument("--epochs",   type=int,   default=config.NUM_EPOCHS)
    parser.add_argument("--lr",       type=float, default=config.LEARNING_RATE)
    parser.add_argument("--loss",     default="label_smoothing",
                        choices=["ce", "label_smoothing", "focal"])
    parser.add_argument("--no-amp",   action="store_true")
    parser.add_argument("--resume",   default=None)
    args = parser.parse_args()

    train(
        model_name=args.model,
        num_epochs=args.epochs,
        lr=args.lr,
        loss_fn=args.loss,
        use_amp=not args.no_amp,
        resume=args.resume,
    )