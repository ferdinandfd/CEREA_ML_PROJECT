# models/gnn.py — GNN multivarié pour la reconstruction de champs météo

import torch
import torch.nn as nn

from config import GNN_HIDDEN_DIM, GNN_NUM_LAYERS

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False


class SparseToGridGNN(nn.Module):
    """
    GNN multivarié : prend les observations aux capteurs et prédit le champ
    complet sur la grille (H, W) par variable.

    Architecture
    ────────────
    1. Encodeur MLP : projette les features des nœuds dans l'espace latent.
    2. Message passing : N couches GATv2 (attention multi-tête).
    3. Décodeur nœud→grille : les embeddings des capteurs sont utilisés pour
       remplir la grille via un MLP + interpolation par plus proche voisin.

    Paramètres
    ----------
    in_features  : nombre de features par nœud (= K, une valeur par variable)
    out_vars     : nombre de variables à reconstruire (K)
    H, W         : dimensions de la grille de sortie
    hidden_dim   : taille de l'espace latent
    num_layers   : nombre de couches de message passing
    """

    def __init__(self, in_features: int, out_vars: int, H: int, W: int,
                 hidden_dim: int = GNN_HIDDEN_DIM, num_layers: int = GNN_NUM_LAYERS):
        super().__init__()

        if not _PYG_AVAILABLE:
            raise ImportError("Installe torch_geometric : pip install torch_geometric")

        self.H = H
        self.W = W
        self.out_vars = out_vars

        # 1. Encodeur
        self.encoder = nn.Sequential(
            nn.Linear(in_features + 2, hidden_dim),   # +2 pour les coordonnées (y,x)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 2. Message passing GATv2
        self.convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # 3. Décodeur : nœud → K valeurs (une par variable)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_vars),
        )

    def forward(self, data):
        """
        Paramètres
        ----------
        data : torch_geometric.data.Data (ou Batch PyG)
            data.x          : (B*N, K)      valeurs normalisées aux capteurs
            data.edge_index : (2, E)        graphe kNN (tous graphes du batch)
            data.pos        : (B*N, 2)      coordonnées normalisées (y,x)
            data.batch      : (B*N,)        indice de graphe par nœud

        Retourne
        --------
        y_pred : (B, K, H, W)  champ prédit normalisé sur la grille complète
        """
        x, edge_index, pos = data.x, data.edge_index, data.pos
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
                else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        # Concatène coordonnées aux features
        h = self.encoder(torch.cat([x, pos], dim=-1))

        # Message passing avec connexions résiduelles
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, edge_index))

        # Prédiction par nœud : (B*N, K)
        node_preds = self.decoder(h)

        # ── Projection capteurs → grille : traite chaque graphe séparément ──
        # (les positions de graphes différents ne doivent pas se mélanger)
        B = int(batch.max().item()) + 1
        preds = []
        for b in range(B):
            idx = (batch == b)
            preds.append(self._scatter_to_grid(node_preds[idx], pos[idx]))

        return torch.stack(preds, dim=0)   # (B, K, H, W)

    def _scatter_to_grid(self, node_preds: torch.Tensor,
                         pos: torch.Tensor) -> torch.Tensor:
        """
        Place les prédictions des capteurs sur la grille par plus proche voisin.

        node_preds : (N_sensors, K)
        pos        : (N_sensors, 2)  — coordonnées normalisées

        Retourne : (K, H, W)
        """
        H, W = self.H, self.W
        K    = self.out_vars
        device = node_preds.device

        # Indices linéaires des capteurs sur la grille
        sensor_row = (pos[:, 0] * H).long().clamp(0, H - 1)
        sensor_col = (pos[:, 1] * W).long().clamp(0, W - 1)
        sensor_lin = sensor_row * W + sensor_col  # (N_sensors,)

        # Grille de coordonnées (H*W, 2) pour kNN
        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32) / H,
            torch.arange(W, device=device, dtype=torch.float32) / W,
            indexing="ij",
        )
        grid_pos = torch.stack([gy.ravel(), gx.ravel()], dim=1)   # (H*W, 2)

        # kNN rapide (1 voisin) via distances euclidiennes
        diffs = grid_pos.unsqueeze(1) - pos.unsqueeze(0)           # (H*W, N, 2)
        dists = (diffs ** 2).sum(-1)                               # (H*W, N)
        nn_idx = dists.argmin(dim=1)                               # (H*W,)

        # Récupère la prédiction du capteur le plus proche pour chaque pixel
        pred_flat = node_preds[nn_idx]                             # (H*W, K)
        y_pred = pred_flat.T.reshape(K, H, W)                     # (K, H, W)

        return y_pred
