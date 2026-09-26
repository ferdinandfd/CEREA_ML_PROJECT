# models/trueunet.py — architecture StrongUNet (univarié et multivarié)
#
# Améliorations :
#   - Connexions résiduelles dans chaque ConvBlock (style ResNet)
#   - Skip connections denses (style U-Net++)
#   - CBAM (Channel + Spatial Attention) après chaque bloc résiduel
#   - BatchNorm + Dropout2d au bottleneck

import torch
import torch.nn as nn


# ── CBAM ──────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Recalibre l'importance de chaque canal (feature map).

    AvgPool + MaxPool → MLP partagé → sigmoid → poids par canal
    reduction : facteur de compression du MLP (défaut 16)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):                          # x : (B, C, H, W)
        avg = x.mean(dim=(2, 3))                   # (B, C)
        mx  = x.amax(dim=(2, 3))                   # (B, C)
        w   = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C)
        return x * w[:, :, None, None]


class SpatialAttention(nn.Module):
    """
    Recalibre l'importance de chaque position spatiale (détection de motifs globaux).

    [AvgPool(C), MaxPool(C)] → concat → Conv 7×7 → sigmoid → poids spatial
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):                          # x : (B, C, H, W)
        avg = x.mean(dim=1, keepdim=True)          # (B, 1, H, W)
        mx  = x.amax(dim=1, keepdim=True)          # (B, 1, H, W)
        w   = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # (B, 1, H, W)
        return x * w


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018).
    Channel Attention → Spatial Attention en séquence.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# ── ConvBlock résiduel + CBAM ─────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Double conv → BN avec connexion résiduelle (ResNet) et CBAM.

    Structure :
        x → Conv→BN→ReLU→Conv→BN → + shortcut(x) → CBAM → ReLU → [Dropout2d]

    Le CBAM recalibre canaux et positions spatiales avant le ReLU final.
    """
    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in,  c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False),
                          nn.BatchNorm2d(c_out))
            if c_in != c_out else nn.Identity()
        )
        self.cbam    = CBAM(c_out)
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        out = self.cbam(self.conv(x) + self.shortcut(x))
        return self.dropout(self.relu(out))


# ── StrongUNet ────────────────────────────────────────────────────────────────

class StrongUNet(nn.Module):
    """
    U-Net à 3 niveaux avec :
      - Connexions résiduelles dans chaque ConvBlock (ResNet)
      - CBAM (Channel + Spatial Attention) après chaque bloc
      - Skip connections denses (U-Net++) :
            d3 ← up(b)  + e3
            d2 ← up(d3) + e2 + pool(e1)
            d1 ← up(d2) + e1 + up(e2)

    Paramètres
    ----------
    in_ch   : 3 (univarié) ou 3*K (multivarié)
    out_ch  : 1 (univarié) ou K   (multivarié)
    base    : canaux de base (×2 à chaque niveau, ×8 au goulot)
    dropout : Dropout2d au bottleneck

    Canaux (base=64) :
        e1 →  64  | pleine résolution  (32×64)
        e2 → 128  | /2                 (16×32)
        e3 → 256  | /4                 ( 8×16)
        b  → 512  | /8 — goulot        ( 4× 8)
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1,
                 base: int = 64, dropout: float = 0.3):
        super().__init__()

        # ── Encodeur ──────────────────────────────────────────────────────────
        self.e1 = ConvBlock(in_ch,    base)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = ConvBlock(base,     base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)

        # ── Goulot ────────────────────────────────────────────────────────────
        self.b  = ConvBlock(base * 4, base * 8, dropout=dropout)

        # ── Décodeur niveau 3 : cat(up(b), e3) → 512+256=768 → 256 ──────────
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)

        # ── Décodeur niveau 2 : cat(up(d3), e2, pool(e1)) → 256+128+64=448 → 128
        self.u2           = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.pool_e1_to_2 = nn.MaxPool2d(2)
        self.d2           = ConvBlock(base * 4 + base, base * 2)

        # ── Décodeur niveau 1 : cat(up(d2), e1, up(e2)) → 128+64+128=320 → 64
        self.u1         = nn.ConvTranspose2d(base * 2, base,     2, stride=2)
        self.up_e2_to_1 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.d1         = ConvBlock(base + base + base * 2, base)

        # ── Sortie ────────────────────────────────────────────────────────────
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        # Encodeur
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        b  = self.b(self.p3(e3))

        # Décodeur niveau 3
        d3 = self.d3(torch.cat([self.u3(b), e3], dim=1))

        # Décodeur niveau 2 : + e1 downsamplé
        e1_down = self.pool_e1_to_2(e1)
        d2 = self.d2(torch.cat([self.u2(d3), e2, e1_down], dim=1))

        # Décodeur niveau 1 : + e2 upsamplé
        e2_up = self.up_e2_to_1(e2)
        d1 = self.d1(torch.cat([self.u1(d2), e1, e2_up], dim=1))

        return self.out(d1)
