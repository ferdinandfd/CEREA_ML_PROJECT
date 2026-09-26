# models/unet.py — architecture TinyUNet (univarié et multivarié)

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in,  c_out, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """
    U-Net léger à 2 niveaux de sous-échantillonnage.

    Paramètres
    ----------
    in_ch  : canaux d'entrée
               - univarié  : 3        (voronoi, mask, sparse)
               - multivarié: 3 * K    (triplet par variable)
    out_ch : canaux de sortie
               - univarié  : 1
               - multivarié: K
    base   : largeur de base (x2 à chaque niveau, x4 au goulot)
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 64):
        super().__init__()
        self.e1 = ConvBlock(in_ch,    base)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = ConvBlock(base,     base * 2)
        self.p2 = nn.MaxPool2d(2)

        self.b  = ConvBlock(base * 2, base * 4)

        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base,     2, stride=2)
        self.d1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        b  = self.b(self.p2(e2))
        d2 = self.d2(torch.cat([self.u2(b), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)
