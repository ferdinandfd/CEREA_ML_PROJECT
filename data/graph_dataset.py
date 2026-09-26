# data/graph_dataset.py — Dataset PyTorch Geometric pour le GNN multivarié

import numpy as np
import torch
from torch.utils.data import Dataset

from config import KEEP_RATIO, GNN_KNN_K
from data.dataset import make_sparse_mask, voronoi_fill_from_sparse

# torch_geometric est importé à la demande pour ne pas casser le reste du projet
# si la lib n'est pas installée.
try:
    from torch_geometric.data import Data
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False
    print("[graph_dataset] torch_geometric non installé — GraphDataset indisponible.")


def build_knn_graph(positions: np.ndarray, k: int = GNN_KNN_K):
    """
    Construit un graphe kNN symétrique à partir de positions (N, 2).
    Retourne edge_index (2, E) en LongTensor.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(positions)
    # k+1 car le point lui-même est inclus dans les voisins
    dists, indices = tree.query(positions, k=k + 1)

    src, dst = [], []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:          # exclut le point lui-même
            src.append(i)
            dst.append(j)
            src.append(j)               # symétrie
            dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # supprime les doublons dus à la symétrisation
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


class MultiGraphDataset(Dataset):
    """
    Dataset PyG pour le GNN multivarié.

    Chaque item est un objet torch_geometric.data.Data contenant :
        x          : (N_sensors, K*2)  features par capteur
                       [valeur_normalisée_k, coord_y_norm, coord_x_norm ... pour chaque var k]
                       En pratique : [val_k pour k in keys, y_norm, x_norm]
        edge_index : (2, E)            graphe kNN sur les capteurs
        pos        : (N_sensors, 2)    coordonnées (row, col) normalisées
        y_grid     : (K, H, W)         champ complet normalisé (supervision)
        mask       : (H, W)            masque binaire partagé
        means      : liste des mu par variable (pour dénormalisation)
        stds       : liste des sigma par variable (pour dénormalisation)

    Remarque : le GNN prédit sur les N_sensors nœuds, le décodage vers la grille
    complète (H, W) est fait dans le modèle ou dans evaluate.py via Voronoï inverse.
    """

    def __init__(self, fields: dict, keys: list[str], means: dict, stds: dict,
                 indices: np.ndarray, keep_ratio: float = KEEP_RATIO,
                 base_seed: int = 0, knn_k: int = GNN_KNN_K):
        if not _PYG_AVAILABLE:
            raise ImportError("Installe torch_geometric : pip install torch_geometric")

        self.fields     = fields
        self.keys       = list(keys)
        self.means      = means
        self.stds       = stds
        self.indices    = np.array(indices, dtype=int)
        self.keep_ratio = keep_ratio
        self.base_seed  = base_seed
        self.knn_k      = knn_k

        any_key = self.keys[0]
        _, self.H, self.W = self.fields[any_key].shape

        # coordonnées de grille normalisées (réutilisées pour toutes les positions)
        gy, gx = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing="ij")
        self._grid_pos = np.stack([gy.ravel() / self.H, gx.ravel() / self.W], axis=1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        t   = int(self.indices[i])
        rng = np.random.default_rng(self.base_seed + t)

        # masque partagé entre variables
        mask = make_sparse_mask(self.H, self.W, self.keep_ratio, rng)
        sensor_idx = np.where(mask.ravel() == 1)[0]       # indices linéaires des capteurs
        N_sensors  = len(sensor_idx)

        # positions des capteurs (normalisées dans [0,1])
        pos = self._grid_pos[sensor_idx]                  # (N_sensors, 2)

        # features des nœuds : valeur normalisée de chaque variable au capteur
        node_feats = []
        y_grids    = []

        for k in self.keys:
            field  = self.fields[k][t]
            mu, sig = self.means[k], self.stds[k]

            obs_norm  = (field.ravel()[sensor_idx] - mu) / sig  # (N_sensors,)
            node_feats.append(obs_norm)

            # grille complète normalisée (supervision)
            y_grids.append(((field - mu) / sig).astype(np.float32))

        # x : (N_sensors, K)  — une feature par variable par nœud
        x = np.stack(node_feats, axis=1).astype(np.float32)

        # graphe kNN sur les capteurs
        edge_index = build_knn_graph(pos, k=self.knn_k)

        y_grid = np.stack(y_grids, axis=0)     # (K, H, W)
        mask2d = mask                           # (H, W)

        data = Data(
            x          = torch.tensor(x, dtype=torch.float32),
            edge_index = edge_index,
            pos        = torch.tensor(pos, dtype=torch.float32),
            y_grid     = torch.tensor(y_grid, dtype=torch.float32),
            mask       = torch.tensor(mask2d, dtype=torch.float32),
        )
        return data
