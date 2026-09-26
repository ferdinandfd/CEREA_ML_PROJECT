# evaluate.py — calcul de RMSE global et sur points non observés
# + NRMSE (normalisée par sigma, comparable entre variables)
# + SSIM (Structural Similarity Index)
# + Robustesse aux capteurs (n_trials masques aléatoires)

import numpy as np
import torch
from torch.utils.data import DataLoader


def ssim_2d(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM entre deux arrays 2D (H,W) en unités physiques."""
    C1 = (0.01 * (gt.max() - gt.min())) ** 2
    C2 = (0.03 * (gt.max() - gt.min())) ** 2
    mu_p, mu_g = pred.mean(), gt.mean()
    sig_p  = pred.std()
    sig_g  = gt.std()
    sig_pg = float(np.mean((pred - mu_p) * (gt - mu_g)))
    num = (2 * mu_p * mu_g + C1) * (2 * sig_pg + C2)
    den = (mu_p**2 + mu_g**2 + C1) * (sig_p**2 + sig_g**2 + C2)
    return float(num / (den + 1e-10))


def evaluate_univariate(model, test_loader, mu: float, sigma: float,
                         device) -> dict:
    """
    Calcule RMSE global et RMSE non-observé pour le modèle univarié.
    Ajoute NRMSE = RMSE / sigma (sans unité).
    Ajoute SSIM moyen sur le batch.

    Retourne
    --------
    {"rmse_global": float, "rmse_unknown": float,
     "nrmse_global": float, "nrmse_unknown": float,
     "ssim": float}
    """
    model.eval()
    sse_all = 0.0
    n_all = 0
    sse_unknown = 0.0
    n_unknown = 0.0
    ssim_scores = []

    with torch.no_grad():
        for x, y, m in test_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x)

            pred_k = pred * sigma + mu
            y_k    = y    * sigma + mu

            e2 = (pred_k - y_k) ** 2
            sse_all += float(e2.sum().item())
            n_all += int(e2.numel())

            unknown = 1.0 - m
            sse_unknown += float((e2 * unknown).sum().item())
            n_unknown += float(unknown.sum().item())

            # SSIM par image dans le batch, puis moyenne
            pred_np = pred_k.squeeze(1).cpu().numpy()  # (B, H, W)
            y_np    = y_k.squeeze(1).cpu().numpy()
            batch_ssim = float(np.mean([
                ssim_2d(pred_np[b], y_np[b]) for b in range(pred_np.shape[0])
            ]))
            ssim_scores.append(batch_ssim)

    rmse_g = float(np.sqrt(sse_all / max(n_all, 1)))
    rmse_u = float(np.sqrt(sse_unknown / (n_unknown + 1e-6)))
    return {
        "rmse_global":   rmse_g,
        "rmse_unknown":  rmse_u,
        "nrmse_global":  rmse_g / sigma,
        "nrmse_unknown": rmse_u / sigma,
        "ssim":          float(np.mean(ssim_scores)),
    }


def evaluate_multivariate(model, test_loader, keys: list[str],
                           means: dict, stds: dict, device) -> dict:
    """
    Calcule RMSE global et RMSE non-observé par variable pour le modèle multivarié.
    Ajoute NRMSE = RMSE / sigma par variable (comparable entre variables).
    Ajoute SSIM par variable.

    Retourne
    --------
    {key: {"rmse_global": float, "rmse_unknown": float,
           "nrmse_global": float, "nrmse_unknown": float,
           "ssim": float}, ...}
    """
    model.eval()
    sse_all     = {k: 0.0 for k in keys}
    n_all       = {k: 0 for k in keys}
    sse_unknown = {k: 0.0 for k in keys}
    n_unknown   = {k: 0.0 for k in keys}
    ssim_all     = {k: [] for k in keys}

    with torch.no_grad():
        for x, y, m in test_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x)

            for i, k in enumerate(keys):
                mu, sig = means[k], stds[k]
                pred_k  = pred[:, i] * sig + mu
                y_k     = y[:, i]    * sig + mu

                e2 = (pred_k - y_k) ** 2
                sse_all[k] += float(e2.sum().item())
                n_all[k] += int(e2.numel())

                unknown = 1.0 - m[:, i]
                sse_unknown[k] += float((e2 * unknown).sum().item())
                n_unknown[k] += float(unknown.sum().item())

                # SSIM par image dans le batch, puis moyenne
                pred_np = pred_k.cpu().numpy()  # (B, H, W)
                y_np    = y_k.cpu().numpy()
                batch_ssim = float(np.mean([
                    ssim_2d(pred_np[b], y_np[b]) for b in range(pred_np.shape[0])
                ]))
                ssim_all[k].append(batch_ssim)

    results = {}
    for k in keys:
        rmse_g = float(np.sqrt(sse_all[k] / max(n_all[k], 1)))
        rmse_u = float(np.sqrt(sse_unknown[k] / (n_unknown[k] + 1e-6)))
        results[k] = {
            "rmse_global":   rmse_g,
            "rmse_unknown":  rmse_u,
            "nrmse_global":  rmse_g / stds[k],
            "nrmse_unknown": rmse_u / stds[k],
            "ssim":          float(np.mean(ssim_all[k])),
        }
    return results


def evaluate_robustness(model, T_test, mu, sigma, device,
                        n_trials=5, keep_ratio=0.05, absolute_indices=None):
    """
    Évalue la robustesse en testant n_trials seeds de masques différents.
    Retourne mean et std des RMSE et SSIM sur les n_trials.

    Parameters
    ----------
    model            : modèle univarié entraîné
    T_test           : tableau numpy (N, H, W) des données de test (dénormalisées ou brutes)
    mu, sigma        : stats de normalisation t2m
    device           : torch device
    n_trials         : nombre de tirages de masques
    keep_ratio       : fraction de capteurs conservés
    absolute_indices : indices temporels absolus associés à T_test

    Retourne
    --------
    {"rmse_mean": float, "rmse_std": float,
     "ssim_mean": float, "ssim_std": float}
    """
    from torch.utils.data import DataLoader

    rmse_trials = []
    ssim_trials = []

    if absolute_indices is None:
        absolute_indices = np.arange(len(T_test), dtype=int)
    else:
        absolute_indices = np.array(absolute_indices, dtype=int)

    for trial in range(n_trials):
        from data.dataset import SparseVoronoiDataset  # import local pour éviter circularité
        ds = SparseVoronoiDataset(T_test, mu=mu, sigma=sigma,
                                   keep_ratio=keep_ratio,
                                   base_seed=trial * 1000,
                                   absolute_indices=absolute_indices)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        res = evaluate_univariate(model, loader, mu, sigma, device)
        rmse_trials.append(res["rmse_global"])
        ssim_trials.append(res["ssim"])

    return {
        "rmse_mean": float(np.mean(rmse_trials)),
        "rmse_std":  float(np.std(rmse_trials)),
        "ssim_mean": float(np.mean(ssim_trials)),
        "ssim_std":  float(np.std(ssim_trials)),
    }


def print_results_univariate(results: dict, var_name: str = "t2m"):
    print(f"\n── Résultats {var_name} ──────────────────────────────")
    print(f"  RMSE global    : {results['rmse_global']:.4f}    NRMSE : {results['nrmse_global']:.4f}")
    print(f"  RMSE non-obs.  : {results['rmse_unknown']:.4f}    NRMSE : {results['nrmse_unknown']:.4f}")
    print(f"  SSIM           : {results['ssim']:.4f}")


def print_results_multivariate(results: dict):
    print("\n── Résultats multivarié ──────────────────────────────────────────────────────────────────")
    print(f"  {'Var':>4s} | {'RMSE global':>12s} | {'NRMSE global':>12s} | {'RMSE non-obs.':>13s} | {'NRMSE non-obs.':>14s} | {'SSIM':>6s}")
    print("  " + "-" * 78)
    for k, v in results.items():
        print(f"  {k:>4s} | {v['rmse_global']:>12.4f} | {v['nrmse_global']:>12.4f} | {v['rmse_unknown']:>13.4f} | {v['nrmse_unknown']:>14.4f} | {v['ssim']:>6.4f}")
    nrmse_global_mean  = float(np.mean([v['nrmse_global']  for v in results.values()]))
    nrmse_unknown_mean = float(np.mean([v['nrmse_unknown'] for v in results.values()]))
    ssim_mean          = float(np.mean([v['ssim']          for v in results.values()]))
    print("  " + "-" * 78)
    print(f"  {'moy.':>4s} | {'':12} | {nrmse_global_mean:>12.4f} | {'':13} | {nrmse_unknown_mean:>14.4f} | {ssim_mean:>6.4f}")
