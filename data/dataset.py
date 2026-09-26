# data/dataset.py — Dataset PyTorch pour UNet (univarié et multivarié)

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree

from config import KEEP_RATIO


# ── Primitives masque / Voronoï ────────────────────────────────────────────

def make_sparse_mask(h: int, w: int, keep_ratio: float = KEEP_RATIO, rng=None) -> np.ndarray:
    """
    Génère un masque binaire (H, W) float32 avec keep_ratio*H*W positions à 1.
    rng : np.random.Generator (reproductible si fourni).
    """
    rng = np.random.default_rng() if rng is None else rng
    n = h * w
    k = max(1, int(round(keep_ratio * n)))
    idx = rng.choice(n, size=k, replace=False)
    m = np.zeros(n, dtype=np.float32)
    m[idx] = 1.0
    return m.reshape(h, w)


def voronoi_fill_from_sparse(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Interpole un champ (H, W) par plus proche voisin à partir des capteurs (mask==1).
    Retourne un array float32 (H, W).
    """
    ys, xs = np.where(mask == 1)
    if len(ys) == 0:
        raise ValueError("Masque vide : aucun capteur.")
    vals = field[ys, xs]
    pts  = np.stack([ys, xs], axis=1)
    gy, gx = np.meshgrid(np.arange(field.shape[0]), np.arange(field.shape[1]), indexing="ij")
    query = np.stack([gy.ravel(), gx.ravel()], axis=1)
    tree = cKDTree(pts)
    _, nn = tree.query(query, k=1)
    return vals[nn].reshape(field.shape).astype(np.float32)


# ── Dataset univarié ───────────────────────────────────────────────────────

class SparseVoronoiDataset(Dataset):
    """
    Dataset pour le modèle UNet univarié (une seule variable).

    Chaque item retourne :
        x : (3, H, W)  [voronoi_norm, mask, sparse_norm]
        y : (1, H, W)  champ normalisé ground truth
        m : (1, H, W)  masque binaire
    """

    def __init__(self, T: np.ndarray, mu: float, sigma: float,
                 keep_ratio: float = KEEP_RATIO, base_seed: int = 0,
                 absolute_indices: np.ndarray | None = None):
        self.T          = T
        self.mu         = mu
        self.sigma      = sigma
        self.keep_ratio = keep_ratio
        self.base_seed  = base_seed
        if absolute_indices is None:
            self.absolute_indices = np.arange(len(T), dtype=int)
        else:
            self.absolute_indices = np.array(absolute_indices, dtype=int)
            if len(self.absolute_indices) != len(T):
                raise ValueError("absolute_indices doit avoir la même longueur que T")

    def __len__(self):
        return len(self.T)

    def __getitem__(self, i: int):
        field = self.T[i]
        t_abs = int(self.absolute_indices[i])
        rng   = np.random.default_rng(self.base_seed + t_abs)
        mask  = make_sparse_mask(field.shape[0], field.shape[1], self.keep_ratio, rng)
        vor   = voronoi_fill_from_sparse(field, mask)
        sparse = field * mask

        x = np.stack([(vor - self.mu) / self.sigma,
                       mask,
                       (sparse - self.mu) / self.sigma], axis=0)   # (3,H,W)
        y = ((field - self.mu) / self.sigma)[None]                  # (1,H,W)
        m = mask[None]                                              # (1,H,W)

        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(m, dtype=torch.float32))


# ── Dataset multivarié ────────────────────────────────────────────────────

class MultiSparseVoronoiDataset(Dataset):
    """
    Dataset pour le modèle UNet multivarié (K variables simultanées).

    Chaque item retourne :
        x : (3*K, H, W)  [voronoi_k, mask_k, sparse_k] pour chaque variable
        y : (K,   H, W)  champs normalisés ground truth
        m : (K,   H, W)  masques binaires (un par variable)

    Paramètres
    ----------
    different_masks : bool
        False (défaut) → même masque spatial pour toutes les variables au temps t.
        True           → masque indépendant par variable.
    """

    def __init__(self, fields: dict, keys: list[str], means: dict, stds: dict,
                 indices: np.ndarray, keep_ratio: float = KEEP_RATIO,
                 base_seed: int = 0, different_masks: bool = False):
        self.fields          = fields
        self.keys            = list(keys)
        self.means           = means
        self.stds            = stds
        self.indices         = np.array(indices, dtype=int)
        self.keep_ratio      = keep_ratio
        self.base_seed       = base_seed
        self.different_masks = different_masks

        any_key = self.keys[0]
        _, self.H, self.W = self.fields[any_key].shape

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        t = int(self.indices[i])
        rng_main    = np.random.default_rng(self.base_seed + t)
        shared_mask = make_sparse_mask(self.H, self.W, self.keep_ratio, rng_main)

        x_ch, y_ch, m_ch = [], [], []

        for vi, k in enumerate(self.keys):
            field = self.fields[k][t]
            mu, sig = self.means[k], self.stds[k]

            if self.different_masks:
                rng_k = np.random.default_rng(self.base_seed + t * 1000 + vi)
                mask  = make_sparse_mask(self.H, self.W, self.keep_ratio, rng_k)
            else:
                mask = shared_mask

            vor    = voronoi_fill_from_sparse(field, mask)
            sparse = field * mask

            x_ch += [(vor - mu) / sig, mask.astype(np.float32), (sparse - mu) / sig]
            y_ch.append(((field - mu) / sig).astype(np.float32))
            m_ch.append(mask.astype(np.float32))

        return (torch.tensor(np.stack(x_ch, axis=0), dtype=torch.float32),
                torch.tensor(np.stack(y_ch, axis=0), dtype=torch.float32),
                torch.tensor(np.stack(m_ch, axis=0), dtype=torch.float32))
