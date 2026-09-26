# train.py — boucles d'entraînement + fonctions de loss

import numpy as np
import torch

from config import WEIGHT_UNKNOWN, WEIGHT_ALL, LAMBDA_GRAD


# ── Gradient loss ──────────────────────────────────────────────────────────

def gradient_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Pénalise les différences de gradients spatiaux entre pred et GT.
    Force la reconstruction des contours et structures fines (fronts, etc.)

    pred, y : (B, C, H, W)
    Retourne un scalaire.
    """
    # Gradients horizontaux (direction W)
    gx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gx_gt   = y[:, :, :, 1:]   - y[:, :, :, :-1]
    # Gradients verticaux (direction H)
    gy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gy_gt   = y[:, :, 1:, :]   - y[:, :, :-1, :]

    return ((gx_pred - gx_gt) ** 2).mean() + ((gy_pred - gy_gt) ** 2).mean()


# ── Fonctions de loss ──────────────────────────────────────────────────────

def loss_fn(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """
    Loss univariée : priorité aux points non observés + gradient loss.
    pred, y, m : (B, 1, H, W)
    """
    unknown   = 1.0 - m
    l_unknown = ((pred - y) ** 2 * unknown).sum() / (unknown.sum() + 1e-6)
    l_all     = ((pred - y) ** 2).mean()
    l_mse     = WEIGHT_UNKNOWN * l_unknown + WEIGHT_ALL * l_all
    return l_mse + LAMBDA_GRAD * gradient_loss(pred, y)


def loss_fn_mv(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor,
               weights: list, device) -> torch.Tensor:
    """
    Loss multivariée pondérée par variable + gradient loss.

    pred, y : (B, K, H, W)
    m       : (B, K, H, W)
    weights : liste de K flottants (mis à jour dynamiquement si adaptatif)
    """
    w    = torch.tensor(weights, dtype=torch.float32, device=device)
    err2 = (pred - y) ** 2 * w[None, :, None, None]

    unknown   = 1.0 - m
    l_unknown = (err2 * unknown).sum() / (unknown.sum() + 1e-6)
    l_all     = err2.mean()
    l_mse     = WEIGHT_UNKNOWN * l_unknown + WEIGHT_ALL * l_all
    return l_mse + LAMBDA_GRAD * gradient_loss(pred, y)


# ── Loss adaptative ────────────────────────────────────────────────────────

def compute_adaptive_weights(nrmse_dict: dict, keys: list) -> list:
    """
    Recalcule les poids à partir des NRMSE courantes (espace normalisé).

    Principe : poids_k = NRMSE_k / mean(NRMSE)
    → les variables mal reconstruites reçoivent plus de poids.
    La somme des poids = K (nombre de variables).

    Paramètres
    ----------
    nrmse_dict : {key: float}  NRMSE par variable sur le split val
    keys       : ordre des variables

    Retourne
    --------
    list de K flottants
    """
    vals  = [nrmse_dict[k] for k in keys]
    total = sum(vals)
    K     = len(keys)
    return [v / total * K for v in vals]


# ── Boucle d'entraînement univariée ───────────────────────────────────────

def train_model(model, train_loader, val_loader, optimizer, loss_fn_callable,
                device, max_epochs: int = 30, patience: int = 5):
    """
    Boucle d'entraînement générique avec early stopping.
    Utilisée pour le modèle univarié (et multivarié sans adaptatif).
    """
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_losses = []
        for x, y, m in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad()
            loss = loss_fn_callable(model(x), y, m)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for x, y, m in val_loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                va_losses.append(loss_fn_callable(model(x), y, m).item())

        tr_mean = float(np.mean(tr_losses))
        va_mean = float(np.mean(va_losses))
        print(f"Epoch {epoch:02d} | train={tr_mean:.5f} | val={va_mean:.5f}", flush=True)

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


# ── Boucle d'entraînement multivariée adaptative ──────────────────────────

def train_model_adaptive(model, train_loader, val_loader, optimizer, keys: list,
                         device, max_epochs: int = 30, patience: int = 5):
    """
    Boucle d'entraînement multivariée avec poids adaptatifs par variable.

    À chaque époque :
      1. Entraînement avec les poids courants
      2. Calcul des NRMSE par variable sur la validation (espace normalisé)
      3. Mise à jour des poids : poids_k ∝ NRMSE_k

    Les poids sont initialisés uniformément (1.0 pour chaque variable).

    Paramètres
    ----------
    keys : liste des noms de variables dans l'ordre des canaux du modèle
    """
    K = len(keys)
    weights   = [1.0] * K          # initialisation uniforme
    best_val, best_state, wait = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):

        # ── Entraînement ───────────────────────────────────────────────────
        model.train()
        tr_losses = []
        for x, y, m in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad()
            loss = loss_fn_mv(model(x), y, m, weights, device)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        # ── Validation + calcul NRMSE par variable ─────────────────────────
        model.eval()
        va_losses  = []
        nrmse_accum = {k: [] for k in keys}

        with torch.no_grad():
            for x, y, m in val_loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                pred = model(x)
                va_losses.append(loss_fn_mv(pred, y, m, weights, device).item())

                # NRMSE en espace normalisé : RMSE_norm = RMSE_reel / sigma
                for i, k in enumerate(keys):
                    nrmse_k = torch.sqrt(((pred[:, i] - y[:, i]) ** 2).mean())
                    nrmse_accum[k].append(nrmse_k.item())

        # ── Mise à jour des poids adaptatifs ───────────────────────────────
        nrmse_mean = {k: float(np.mean(nrmse_accum[k])) for k in keys}
        weights    = compute_adaptive_weights(nrmse_mean, keys)

        tr_mean = float(np.mean(tr_losses))
        va_mean = float(np.mean(va_losses))
        w_str   = " | ".join(f"{k}={weights[i]:.3f}" for i, k in enumerate(keys))
        print(f"Epoch {epoch:02d} | train={tr_mean:.5f} | val={va_mean:.5f} | poids [{w_str}]",
              flush=True)

        # ── Early stopping ─────────────────────────────────────────────────
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
