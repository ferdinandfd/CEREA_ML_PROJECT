# data/loader.py — chargement ERA5 depuis Google Cloud Storage

import numpy as np
import xarray as xr

from config import ERA5_PATH, VAR_CANDIDATES, TIME_SLICE, LEVEL_TARGET


def open_dataset() -> xr.Dataset:
    """Ouvre le dataset ERA5 en lecture seule (accès anonyme GCS)."""
    ds = xr.open_zarr(ERA5_PATH, storage_options={"token": "anon"})
    return ds


def pick_first_existing(candidates: list[str], ds: xr.Dataset) -> str | None:
    """Retourne le premier nom de variable trouvé dans ds, ou None."""
    for c in candidates:
        if c in ds.data_vars:
            return c
    return None


def find_variables(ds: xr.Dataset, keys: list[str] | None = None) -> dict[str, str | None]:
    """
    Retourne un dict {clé_courte: nom_ERA5} pour les variables demandées.
    Si keys=None, cherche toutes les variables de VAR_CANDIDATES.
    """
    if keys is None:
        keys = list(VAR_CANDIDATES.keys())
    return {k: pick_first_existing(VAR_CANDIDATES[k], ds) for k in keys}


def to_THW(
    da: xr.DataArray,
    level_target: int = LEVEL_TARGET,
    time_slice: slice = TIME_SLICE,
) -> np.ndarray:
    """
    Extrait une DataArray ERA5 et retourne un array float32 (T, H, W).
    Gère les variables 3D (time, lat, lon) et 4D (time, level, lat, lon).
    """
    if "level" in da.dims:
        da = da.sel(level=level_target, method="nearest")

    if "time" not in da.dims:
        raise ValueError(f"{da.name}: pas de dimension time. dims={da.dims}")
    da = da.isel(time=time_slice)

    lat_dim = next((d for d in da.dims if d in ("lat", "latitude")), None)
    lon_dim = next((d for d in da.dims if d in ("lon", "longitude")), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError(f"{da.name}: dimension lat/lon introuvable. dims={da.dims}")

    da = da.transpose("time", lat_dim, lon_dim)
    arr = da.astype("float32").values

    if arr.ndim != 3:
        raise ValueError(f"{da.name}: attendu 3D, obtenu shape={arr.shape}")
    return arr


def load_fields(ds: xr.Dataset, keys: list[str]) -> dict[str, np.ndarray]:
    """
    Charge toutes les variables demandées et retourne un dict {clé: array (T,H,W)}.
    Ignore les variables absentes du dataset.
    """
    vars_found = find_variables(ds, keys)
    fields = {}
    for k, vname in vars_found.items():
        if vname is None:
            print(f"[loader] Variable '{k}' introuvable, ignorée.")
            continue
        fields[k] = to_THW(ds[vname])
        print(f"[loader] {k} ({vname}): {fields[k].shape}")
    return fields


def compute_split_indices(n: int, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """Retourne (idx_train, idx_val, idx_test) sous forme de np.ndarray."""
    n_train = int(train_ratio * n)
    n_val   = int(val_ratio * n)
    idx_train = np.arange(0, n_train)
    idx_val   = np.arange(n_train, n_train + n_val)
    idx_test  = np.arange(n_train + n_val, n)
    return idx_train, idx_val, idx_test


def compute_stats(fields: dict[str, np.ndarray], idx_train: np.ndarray):
    """Calcule moyenne et écart-type sur le split train uniquement."""
    means, stds = {}, {}
    for k, arr in fields.items():
        means[k] = float(arr[idx_train].mean())
        stds[k]  = float(arr[idx_train].std() + 1e-6)
        print(f"[loader] {k}: mu={means[k]:.4f}, sigma={stds[k]:.4f}")
    return means, stds
