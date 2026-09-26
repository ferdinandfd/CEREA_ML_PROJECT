# run_gnn_multivariate.py — pipeline complet modèle GNN multivarié

import numpy as np
import torch
from torch_geometric.loader import DataLoader

import config
from data.loader import open_dataset, load_fields, compute_split_indices, compute_stats
from data.graph_dataset import MultiGraphDataset
from models.gnn import SparseToGridGNN
from evaluate import print_results_multivariate


def loss_fn_gnn(pred: torch.Tensor, y_grid: torch.Tensor,
                mask: torch.Tensor, weights: list, device) -> torch.Tensor:
    """
    Loss multivariée pour le GNN avec poids adaptatifs + gradient loss.
    weights : liste de K flottants (adaptatifs ou MULTIVAR_WEIGHTS)
    """
    from train import gradient_loss
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    B, K, H, W = pred.shape
    y_grid = y_grid.view(B, K, H, W)
    mask   = mask.view(B, H, W)
    err2   = (pred - y_grid) ** 2 * w[None, :, None, None]
    unknown   = (1.0 - mask).unsqueeze(1)
    l_unknown = (err2 * unknown).sum() / (unknown.sum() * K + 1e-6)
    l_all     = err2.mean()
    l_mse     = config.WEIGHT_UNKNOWN * l_unknown + config.WEIGHT_ALL * l_all
    return l_mse + config.LAMBDA_GRAD * gradient_loss(pred, y_grid)


def evaluate_gnn(model, loader, keys: list[str], means: dict, stds: dict, device):
    model.eval()
    rmse_all     = {k: [] for k in keys}
    rmse_unknown = {k: [] for k in keys}

    with torch.no_grad():
        for data in loader:
            data   = data.to(device)
            pred   = model(data)          # (B, K, H, W)
            B, K, H, W = pred.shape
            y_grid = data.y_grid.view(B, K, H, W)
            mask   = data.mask.view(B, H, W)

            for i, k in enumerate(keys):
                mu, sig = means[k], stds[k]
                pred_k  = pred[:, i] * sig + mu
                y_k     = y_grid[:, i] * sig + mu

                e2 = (pred_k - y_k) ** 2
                rmse_all[k].append(torch.sqrt(e2.mean()).item())

                unknown = 1.0 - mask
                e2u = (e2 * unknown).sum() / (unknown.sum() + 1e-6)
                rmse_unknown[k].append(torch.sqrt(e2u).item())

    return {
        k: {
            "rmse_global":  float(np.mean(rmse_all[k])),
            "rmse_unknown": float(np.mean(rmse_unknown[k])),
        }
        for k in keys
    }


def train_gnn_adaptive(model, train_loader, val_loader, optimizer, keys: list,
                        device, max_epochs: int = 30, patience: int = 5):
    """
    Entraînement GNN avec poids adaptatifs par variable + gradient loss.
    Poids initialisés uniformément, recalculés à chaque époque selon NRMSE val.
    """
    from train import compute_adaptive_weights
    K       = len(keys)
    weights = [1.0] * K
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_losses = []
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn_gnn(pred, data.y_grid, data.mask, weights, device)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses   = []
        nrmse_accum = {k: [] for k in keys}
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred = model(data)
                va_losses.append(loss_fn_gnn(pred, data.y_grid, data.mask, weights, device).item())
                B, K_out, H, W = pred.shape
                y_grid = data.y_grid.view(B, K_out, H, W)
                for i, k in enumerate(keys):
                    nrmse_k = torch.sqrt(((pred[:, i] - y_grid[:, i]) ** 2).mean())
                    nrmse_accum[k].append(nrmse_k.item())

        nrmse_mean = {k: float(np.mean(nrmse_accum[k])) for k in keys}
        weights    = compute_adaptive_weights(nrmse_mean, keys)

        tr_mean = float(np.mean(tr_losses))
        va_mean = float(np.mean(va_losses))
        w_str   = " | ".join(f"{k}={weights[i]:.3f}" for i, k in enumerate(keys))
        print(f"Epoch {epoch:02d} | train={tr_mean:.5f} | val={va_mean:.5f} | poids [{w_str}]",
              flush=True)

        if va_mean < best_val - 1e-5:
            best_val   = va_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping à l'epoch {epoch}.", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"Meilleur val loss : {best_val:.5f}", flush=True)
    return model


def main():
    # ── 1. Données ─────────────────────────────────────────────────────────
    print("Chargement ERA5…")
    ds     = open_dataset()
    fields = load_fields(ds, config.MULTIVAR_KEYS)
    keys   = list(fields.keys())

    # ── 2. Split + stats ────────────────────────────────────────────────────
    N = next(iter(fields.values())).shape[0]
    idx_train, idx_val, idx_test = compute_split_indices(N)
    means, stds = compute_stats(fields, idx_train)

    any_key = keys[0]
    _, H, W = fields[any_key].shape

    # ── 3. Datasets & DataLoaders ───────────────────────────────────────────
    train_ds = MultiGraphDataset(
        fields, keys, means, stds, idx_train,
        keep_ratio=config.KEEP_RATIO, base_seed=config.SEED_TRAIN,
        knn_k=config.GNN_KNN_K,
    )
    val_ds = MultiGraphDataset(
        fields, keys, means, stds, idx_val,
        keep_ratio=config.KEEP_RATIO, base_seed=config.SEED_VAL,
        knn_k=config.GNN_KNN_K,
    )
    test_ds = MultiGraphDataset(
        fields, keys, means, stds, idx_test,
        keep_ratio=config.KEEP_RATIO, base_seed=config.SEED_TEST,
        knn_k=config.GNN_KNN_K,
    )

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE_EVAL,  shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE_EVAL,  shuffle=False)

    # ── 4. Modèle ───────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K      = len(keys)

    model_gnn = SparseToGridGNN(
        in_features=K,
        out_vars=K,
        H=H,
        W=W,
        hidden_dim=config.GNN_HIDDEN_DIM,
        num_layers=config.GNN_NUM_LAYERS,
    ).to(device)

    opt_gnn = torch.optim.Adam(model_gnn.parameters(), lr=config.LR)
    print(f"Modèle GNN (K={K}, hidden={config.GNN_HIDDEN_DIM}, layers={config.GNN_NUM_LAYERS}) sur {device}")

    # ── 5. Entraînement adaptatif ────────────────────────────────────────────
    model_gnn = train_gnn_adaptive(
        model_gnn, train_loader, val_loader, opt_gnn,
        keys=keys,
        device=device,
        max_epochs=config.MAX_EPOCHS,
        patience=config.PATIENCE,
    )

    # ── 6. Évaluation ───────────────────────────────────────────────────────
    results = evaluate_gnn(model_gnn, test_loader, keys, means, stds, device)
    print_results_multivariate(results)


if __name__ == "__main__":
    main()
