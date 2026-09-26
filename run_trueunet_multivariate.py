# run_trueunet_multivariate.py — pipeline StrongUNet multivarié

import torch
from torch.utils.data import DataLoader

import config
from data.loader import open_dataset, load_fields, compute_split_indices, compute_stats
from data.dataset import MultiSparseVoronoiDataset
from models.trueunet import StrongUNet
from train import train_model_adaptive
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
    model_mv = StrongUNet(in_ch=3 * K, out_ch=K,
                          base=config.UNET_BASE, dropout=0.3).to(device)
    opt_mv   = torch.optim.Adam(model_mv.parameters(), lr=config.LR)

    n_params = sum(p.numel() for p in model_mv.parameters())
    print(f"Modèle StrongUNet multivarié (K={K}, base={config.UNET_BASE}, {n_params:,} params) sur {device}")

    # ── 5. Entraînement adaptatif ────────────────────────────────────────────
    model_mv = train_model_adaptive(
        model_mv, train_loader, val_loader, opt_mv,
        keys=keys,
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
