# visualize.py — visualisations GT / Voronoï / prédiction / erreur

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_voronoi_overview(T: np.ndarray, mask: np.ndarray, T_voronoi: np.ndarray):
    """Affiche ground truth / observations 5% / interpolation Voronoï."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    im0 = ax[0].imshow(T, cmap="coolwarm")
    ax[0].set_title("Ground truth T2m")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(np.where(mask == 1, T, np.nan), cmap="coolwarm")
    ax[1].set_title("Observations 5%")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(T_voronoi, cmap="coolwarm")
    ax[2].set_title("Interpolation Voronoï")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()


def plot_univariate_sample(model, dataset, idx: int, mu: float, sigma: float, device):
    """Affiche Original / Observations / Voronoï / Prédiction CNN / Erreur."""
    model.eval()
    x, y, m = dataset[idx]

    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).cpu().squeeze(0)

    y_k     = (y.numpy()     * sigma + mu).squeeze(0)
    vor_k   = (x.numpy()[0]  * sigma + mu)
    pred_k  = (pred.numpy()  * sigma + mu).squeeze(0)
    mask_np = m.numpy().squeeze(0)
    sparse_k = y_k * mask_np

    err_vor = np.sqrt(np.mean((vor_k  - y_k) ** 2))
    err_cnn = np.sqrt(np.mean((pred_k - y_k) ** 2))

    vmin = min(y_k.min(), vor_k.min(), pred_k.min())
    vmax = max(y_k.max(), vor_k.max(), pred_k.max())

    fig, ax = plt.subplots(1, 5, figsize=(22, 4))

    ax[0].imshow(y_k,    cmap="coolwarm", vmin=vmin, vmax=vmax); ax[0].set_title("Original T2m")
    ax[1].imshow(sparse_k, cmap="coolwarm", vmin=vmin, vmax=vmax); ax[1].set_title("Observations 5%")
    ax[2].imshow(vor_k,  cmap="coolwarm", vmin=vmin, vmax=vmax); ax[2].set_title(f"Voronoï\nRMSE={err_vor:.3f} K")
    ax[3].imshow(pred_k, cmap="coolwarm", vmin=vmin, vmax=vmax); ax[3].set_title(f"CNN\nRMSE={err_cnn:.3f} K")

    im_err = ax[4].imshow(np.abs(pred_k - y_k), cmap="magma")
    ax[4].set_title("|Erreur CNN|")

    fig.colorbar(ax[3].images[0], ax=ax[:4], fraction=0.5, pad=0.01)
    fig.colorbar(im_err, ax=ax[4], fraction=0.046, pad=0.04)

    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()


def plot_multivariate_sample(model, dataset, keys: list[str], means: dict,
                              stds: dict, idx: int, device):
    """Affiche GT / Observations / Voronoï / Prédiction / Erreur pour chaque variable."""
    model.eval()
    x, y, m = dataset[idx]

    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).cpu().squeeze(0)

    fig, axes = plt.subplots(len(keys), 5, figsize=(20, 3 * len(keys)))
    if len(keys) == 1:
        axes = np.array([axes])

    for i, k in enumerate(keys):
        mu, sig = means[k], stds[k]

        gt   = y[i].numpy() * sig + mu
        mask = m[i].numpy()
        obs  = gt * mask
        vor  = x[3 * i].numpy() * sig + mu
        pr   = pred[i].numpy() * sig + mu
        err  = np.sqrt((pr - gt) ** 2)
        err_vor = float(np.sqrt(np.mean((vor - gt) ** 2)))
        err_pr  = float(np.sqrt(np.mean((pr - gt) ** 2)))

        vmin, vmax = gt.min(), gt.max()
        vmax_err   = np.percentile(err, 99)

        im0 = axes[i, 0].imshow(gt,  cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"{k} GT")
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)

        im1 = axes[i, 1].imshow(obs, cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"{k} Obs 5%")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

        im2 = axes[i, 2].imshow(vor, cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[i, 2].set_title(f"{k} Voronoï\nRMSE={err_vor:.3f}")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

        im3 = axes[i, 3].imshow(pr,  cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[i, 3].set_title(f"{k} Pred\nRMSE={err_pr:.3f}")
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)

        im4 = axes[i, 4].imshow(err, cmap="magma", vmin=0.0, vmax=vmax_err)
        axes[i, 4].set_title(f"{k} |Erreur|")
        plt.colorbar(im4, ax=axes[i, 4], fraction=0.046)

    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_gnn_sample(model, dataset, keys: list[str], means: dict,
                    stds: dict, idx: int, device):
    """Affiche GT / Capteurs 5% / Prédiction GNN / Erreur pour chaque variable."""
    model.eval()
    data = dataset[idx]

    with torch.no_grad():
        pred = model(data.to(device)).cpu().squeeze(0)

    y = data.y_grid.cpu()
    H, W = int(data.mask.shape[0]), int(data.mask.shape[1])
    pos = data.pos.cpu().numpy()
    iy = np.clip(np.round(pos[:, 0] * H).astype(int), 0, H - 1)
    ix = np.clip(np.round(pos[:, 1] * W).astype(int), 0, W - 1)

    fig, axes = plt.subplots(len(keys), 4, figsize=(16, 3 * len(keys)))
    if len(keys) == 1:
        axes = np.array([axes])

    for i, k in enumerate(keys):
        mu_k, sig_k = means[k], stds[k]
        gt = y[i].numpy() * sig_k + mu_k
        pr = pred[i].numpy() * sig_k + mu_k

        sensors = np.full((H, W), np.nan, dtype=np.float32)
        sensor_vals = data.x[:, i].cpu().numpy() * sig_k + mu_k
        sensors[iy, ix] = sensor_vals

        err = np.abs(pr - gt)
        vmin, vmax = float(gt.min()), float(gt.max())
        vmax_err = float(np.percentile(err, 99))

        im0 = axes[i, 0].imshow(gt, cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"{k} GT")
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)

        im1 = axes[i, 1].imshow(np.ma.masked_invalid(sensors), cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"{k} Capteurs (5%)")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

        im2 = axes[i, 2].imshow(pr, cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[i, 2].set_title(f"{k} Pred GNN")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

        im3 = axes[i, 3].imshow(err, cmap="magma", vmin=0.0, vmax=vmax_err)
        axes[i, 3].set_title(f"{k} Erreur")
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046)

    for a in axes.ravel():
        a.axis("off")
    plt.tight_layout()
    plt.show()
