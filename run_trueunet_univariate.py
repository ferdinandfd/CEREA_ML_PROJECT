# run_trueunet_univariate.py — pipeline StrongUNet univarié (T2m)

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data.loader import open_dataset, compute_split_indices
from data.dataset import SparseVoronoiDataset
from models.trueunet import StrongUNet
from train import loss_fn, train_model
from evaluate import evaluate_univariate, print_results_univariate
from visualize import plot_univariate_sample


def main():
    # ── 1. Données ─────────────────────────────────────────────────────────
    print("Chargement ERA5…")
    ds = open_dataset()

    VAR = next((v for v in ["2m_temperature", "t2m"] if v in ds.data_vars), None)
    if VAR is None:
        raise ValueError("Variable T2m introuvable dans le dataset.")

    N_max    = min(2000, ds[VAR].time.size)
    T_all    = ds[VAR].isel(time=slice(0, N_max)).values.astype(np.float32)
    good     = np.isfinite(T_all).all(axis=(1, 2))
    T_all    = T_all[good]
    print(f"T_all chargé : {T_all.shape}")

    # ── 2. Split + normalisation ────────────────────────────────────────────
    idx_train, idx_val, idx_test = compute_split_indices(len(T_all))
    T_train = T_all[idx_train]
    T_val   = T_all[idx_val]
    T_test  = T_all[idx_test]

    mu    = float(T_train.mean())
    sigma = float(T_train.std() + 1e-6)
    print(f"Normalisation : mu={mu:.4f}, sigma={sigma:.4f}")

    # ── 3. Datasets & DataLoaders ───────────────────────────────────────────
    train_ds = SparseVoronoiDataset(T_train, mu, sigma,
                                    keep_ratio=config.KEEP_RATIO,
                                    base_seed=config.SEED_TRAIN,
                                    absolute_indices=idx_train)
    val_ds   = SparseVoronoiDataset(T_val,   mu, sigma,
                                    keep_ratio=config.KEEP_RATIO,
                                    base_seed=config.SEED_VAL,
                                    absolute_indices=idx_val)
    test_ds  = SparseVoronoiDataset(T_test,  mu, sigma,
                                    keep_ratio=config.KEEP_RATIO,
                                    base_seed=config.SEED_TEST,
                                    absolute_indices=idx_test)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE_TRAIN,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE_EVAL,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE_EVAL,
                              shuffle=False, num_workers=0)

    # ── 4. Modèle ───────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = StrongUNet(in_ch=3, out_ch=1,
                        base=config.UNET_BASE, dropout=0.3).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=config.LR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle StrongUNet univarié (base={config.UNET_BASE}, {n_params:,} params) sur {device}")

    # ── 5. Entraînement ─────────────────────────────────────────────────────
    model = train_model(
        model, train_loader, val_loader, opt,
        loss_fn_callable=loss_fn,
        device=device,
        max_epochs=config.MAX_EPOCHS,
        patience=config.PATIENCE,
    )

    # ── 6. Évaluation ───────────────────────────────────────────────────────
    results = evaluate_univariate(model, test_loader, mu, sigma, device)
    print_results_univariate(results, var_name="t2m")

    # ── 7. Visualisation ────────────────────────────────────────────────────
    plot_univariate_sample(model, test_ds, idx=0, mu=mu, sigma=sigma, device=device)


if __name__ == "__main__":
    main()
