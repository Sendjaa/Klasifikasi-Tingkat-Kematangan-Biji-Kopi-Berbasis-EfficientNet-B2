# ============================================================
#  model.py — Arsitektur Model Deep Learning (Transfer Learning)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import config


# ── Attention Module (CBAM-lite) ───────────────────────────
class ChannelAttention(nn.Module):
    """Channel Attention untuk menekankan fitur warna biji kopi."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        scale = torch.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * scale


# ── Classifier Head ────────────────────────────────────────
class CoffeeClassifierHead(nn.Module):
    """
    Custom head yang menggantikan classifier default dari backbone.
    Menggunakan MLP dengan BatchNorm dan Dropout.
    """
    def __init__(self, in_features: int, num_classes: int,
                 dropout: float = 0.4):
        super().__init__()
        mid = in_features // 2
        self.head = nn.Sequential(
            nn.Linear(in_features, mid),
            nn.BatchNorm1d(mid),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(mid, mid // 2),
            nn.BatchNorm1d(mid // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(mid // 2, num_classes),
        )

    def forward(self, x):
        return self.head(x)


# ── Model Utama ────────────────────────────────────────────
class CoffeeRoastDetector(nn.Module):
    """
    Model deteksi tingkat penyangraian kopi berbasis Transfer Learning.

    Backbone : EfficientNet-B2 (default) dari timm library
    Head     : Custom MLP classifier
    Extras   : GeM Pooling + Channel Attention (opsional)
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = config.PRETRAINED,
        dropout: float = config.DROPOUT_RATE,
        freeze_layers: int = config.FREEZE_LAYERS,
    ):
        super().__init__()
        self.model_name  = model_name
        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # hapus head bawaan
            global_pool="avg",      # GeM bisa diganti di sini
        )
        in_features = self.backbone.num_features

        # ── Freeze layers (opsional) ──────────────────────
        if freeze_layers > 0:
            self._freeze_early_layers(freeze_layers)

        # ── Custom Head ───────────────────────────────────
        self.classifier = CoffeeClassifierHead(
            in_features, num_classes, dropout
        )

    def _freeze_early_layers(self, n: int):
        params = list(self.backbone.parameters())
        for p in params[:n]:
            p.requires_grad = False
        frozen = sum(1 for p in self.backbone.parameters()
                     if not p.requires_grad)
        total  = len(list(self.backbone.parameters()))
        print(f"[Backbone] {frozen}/{total} parameter dibekukan.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, in_features)
        logits   = self.classifier(features)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Mengembalikan probabilitas kelas (softmax)."""
        with torch.no_grad():
            logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def get_feature_maps(self, x: torch.Tensor):
        """Ekstraksi feature map untuk Grad-CAM."""
        return self.backbone.forward_features(x)

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return {"total": total, "trainable": trainable,
                "frozen": total - trainable}

    def __repr__(self):
        info = self.count_parameters()
        return (f"CoffeeRoastDetector(\n"
                f"  backbone    = {self.model_name}\n"
                f"  num_classes = {self.num_classes}\n"
                f"  params      = {info['total']:,} total | "
                f"{info['trainable']:,} trainable\n)")


# ── Factory ────────────────────────────────────────────────
def build_model(device: str = config.DEVICE, **kwargs) -> CoffeeRoastDetector:
    model = CoffeeRoastDetector(**kwargs)
    model = model.to(device)
    print(model)
    return model


def load_model(path: str, device: str = config.DEVICE,
               **kwargs) -> CoffeeRoastDetector:
    model = CoffeeRoastDetector(**kwargs).to(device)
    state = torch.load(path, map_location=device)
    # Mendukung checkpoint yang menyimpan selain state_dict
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"[Model] Berhasil dimuat dari: {path}")
    return model


# ── Label Smoothing Loss ───────────────────────────────────
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy dengan label smoothing untuk mencegah overconfidence.
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        n = pred.size(-1)
        log_prob = F.log_softmax(pred, dim=-1)
        smooth_val = self.smoothing / (n - 1)
        one_hot = torch.full_like(log_prob, smooth_val)
        one_hot.scatter_(-1, target.unsqueeze(-1),
                         1.0 - self.smoothing)
        return -(one_hot * log_prob).sum(dim=-1).mean()


# ── Focal Loss ────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss — efektif untuk data yang tidak seimbang.
    γ=2 (gamma) adalah nilai yang umum digunakan.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(4, 3, 224, 224).to(config.DEVICE)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Probabilitas: {model.predict_proba(dummy)[0]}")