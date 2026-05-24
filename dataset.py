# ============================================================
#  dataset.py — Dataset & Augmentasi Biji Kopi
# ============================================================

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


# ── Augmentasi Albumentations ──────────────────────────────
def get_transforms(split: str = "train"):
    """
    Mengembalikan pipeline augmentasi sesuai split.
    split: 'train' | 'val' | 'test'
    """
    if split == "train":
        return A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.4),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=20, p=0.5),
            # Augmentasi warna — penting untuk biji kopi (variasi warna sangrai)
            A.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.2, hue=0.05, p=0.6),
            A.HueSaturationValue(hue_shift_limit=10,
                                 sat_shift_limit=20,
                                 val_shift_limit=15, p=0.4),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            # Noise & blur
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
            A.OneOf([
                A.GridDistortion(p=0.5),
                A.ElasticTransform(p=0.5),
                A.OpticalDistortion(p=0.5),
            ], p=0.2),
            # Cutout / coarse dropout
            A.CoarseDropout(max_holes=8, max_height=16,
                            max_width=16, p=0.3),
            A.Normalize(mean=config.MEAN, std=config.STD),
            ToTensorV2(),
        ])
    else:   # val / test
        return A.Compose([
            A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
            A.Normalize(mean=config.MEAN, std=config.STD),
            ToTensorV2(),
        ])


# ── Dataset ────────────────────────────────────────────────
class CoffeeRoastDataset(Dataset):
    """
    Dataset biji kopi berdasarkan struktur folder:
      data/
        train/
          light/       ← gambar biji kopi light roast
          medium/
          medium_dark/
          dark/
        val/  ...
        test/ ...
    """
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, root_dir: str | Path, split: str = "train",
                 transform=None):
        self.root_dir  = Path(root_dir)
        self.split     = split
        self.transform = transform or get_transforms(split)
        self.classes   = config.CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self):
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                continue
            for f in cls_dir.iterdir():
                if f.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((f, self.class_to_idx[cls]))

        if not self.samples:
            print(f"[PERINGATAN] Tidak ada gambar ditemukan di: {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(str(path))
        if image is None:
            image = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3),
                             dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        return augmented["image"], label

    def get_class_weights(self) -> torch.Tensor:
        """Hitung bobot kelas untuk menangani ketidakseimbangan data."""
        counts = [0] * config.NUM_CLASSES
        for _, lbl in self.samples:
            counts[lbl] += 1
        total = sum(counts)
        weights = [total / (config.NUM_CLASSES * c) if c > 0 else 0.0
                   for c in counts]
        return torch.FloatTensor(weights)

    def get_sampler(self) -> WeightedRandomSampler:
        """WeightedRandomSampler untuk oversampling kelas minoritas."""
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[lbl] for _, lbl in self.samples]
        return WeightedRandomSampler(sample_weights, len(sample_weights),
                                     replacement=True)

    def __repr__(self):
        counts = {}
        for _, lbl in self.samples:
            nm = self.classes[lbl]
            counts[nm] = counts.get(nm, 0) + 1
        info = "\n  ".join(f"{k}: {v}" for k, v in counts.items())
        return (f"CoffeeRoastDataset [{self.split}]\n"
                f"  Total: {len(self.samples)} gambar\n  {info}")


# ── DataLoader Factories ───────────────────────────────────
def get_dataloaders(use_weighted_sampler: bool = True) -> dict:
    """
    Membuat DataLoader untuk train / val / test.
    Mengembalikan dict: {'train': ..., 'val': ..., 'test': ...}
    """
    loaders = {}
    for split, root in [
        ("train", config.TRAIN_DIR),
        ("val",   config.VAL_DIR),
        ("test",  config.TEST_DIR),
    ]:
        ds = CoffeeRoastDataset(root, split=split)
        print(ds)

        if split == "train" and use_weighted_sampler and len(ds) > 0:
            sampler = ds.get_sampler()
            loader = DataLoader(ds, batch_size=config.BATCH_SIZE,
                                sampler=sampler, num_workers=4,
                                pin_memory=True, drop_last=True)
        else:
            loader = DataLoader(ds, batch_size=config.BATCH_SIZE,
                                shuffle=(split == "train"),
                                num_workers=4, pin_memory=True)
        loaders[split] = loader

    return loaders


# ── Preview gambar (opsional) ──────────────────────────────
def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Balikkan normalisasi untuk visualisasi."""
    mean = np.array(config.MEAN)
    std  = np.array(config.STD)
    img  = tensor.permute(1, 2, 0).numpy()
    img  = (img * std + mean).clip(0, 1)
    return (img * 255).astype(np.uint8)


if __name__ == "__main__":
    loaders = get_dataloaders()
    for split, loader in loaders.items():
        if len(loader) > 0:
            imgs, lbls = next(iter(loader))
            print(f"[{split}] batch shape: {imgs.shape}, labels: {lbls[:8]}")