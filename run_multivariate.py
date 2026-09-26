# run_multivariate.py — pipeline complet modèle UNet multivarié

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data.loader import open_dataset, load_fields, compute_split_indices, compute_stats
from data.dataset import MultiSparseVoronoiDataset
from models.unet import TinyUNet
from train import loss_fn_mv, train_model
from evaluate import evaluate_multivariate, print_results_multivariate
from visualize import plot_multivariate_sample


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

    # ── 3. Datasets & DataLoaders ───────────────────────────────────────────
    train_mv = MultiSparseVoronoiDataset(
        fields, keys, means, stds, idx_train,
        keep_ratio=config.KEEP_RATIO, base_seed=config.SEED_TRAIN,
    )
    val_mv = MultiSparseVoronoiDataset(
        fields, keys, means, stds, idx_val,
        keep_ratio=config.KEEP_RATIO, base_seed=config.SEED_VAL,
    )
    test_mv = MultiSparseVoronoiDataset(
        fields, keys, means, stds, idx_test,
        keep_ratio=config.KEEP_RATIO, base_seed=config.SEED_TEST,
    )

    train_loader = DataLoader(train_mv, batch_size=config.BATCH_SIZE_TRAIN,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_mv,   batch_size=config.BATCH_SIZE_EVAL,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_mv,  batch_size=config.BATCH_SIZE_EVAL,
                              shuffle=False, num_workers=0)

    # ── 4. Modèle ───────────────────────────────────────────────────────────
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    K        = len(keys)
    model_mv = TinyUNet(in_ch=3 * K, out_ch=K, base=config.UNET_BASE).to(device)
    opt_mv   = torch.optim.Adam(model_mv.parameters(), lr=config.LR)
    print(f"Modèle UNet multivarié (K={K}, base={config.UNET_BASE}) sur {device}")

    # ── 5. Entraînement ─────────────────────────────────────────────────────
    def _loss(pred, y, m):
        return loss_fn_mv(pred, y, m, device=device)

    model_mv = train_model(
        model_mv, train_loader, val_loader, opt_mv,
        loss_fn_callable=_loss,
        device=device,
        max_epochs=config.MAX_EPOCHS,
        patience=config.PATIENCE,
    )

    # ── 6. Évaluation ───────────────────────────────────────────────────────
    results = evaluate_multivariate(model_mv, test_loader, keys, means, stds, device)
    print_results_multivariate(results)

    # ── 7. Visualisation ────────────────────────────────────────────────────
    plot_multivariate_sample(model_mv, test_mv, keys, means, stds, idx=0, device=device)


if __name__ == "__main__":
    main()
