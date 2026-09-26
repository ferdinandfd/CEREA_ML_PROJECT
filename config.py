# config.py — hyperparamètres et constantes globales

# ── Données ────────────────────────────────────────────────────────────────
ERA5_PATH = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"
TIME_SLICE = slice(0, 2000)      # nombre de pas de temps à charger
LEVEL_TARGET = 850               # niveau de pression (hPa) pour u, v, z, q

# Variables à charger (noms candidats dans ERA5)
VAR_CANDIDATES = {
    "u":   ["10m_u_component_of_wind", "u10"],
    "v":   ["10m_v_component_of_wind", "v10"],
    "z":   ["geopotential", "z"],
    "q":   ["specific_humidity", "q"],
    "t2m": ["2m_temperature", "t2m"],
}

# Sous-ensemble de variables utilisé en multivarié
MULTIVAR_KEYS = ["z", "q", "t2m", "u", "v"]

# ── Capteurs / masque ──────────────────────────────────────────────────────
KEEP_RATIO = 0.05                # fraction de pixels observés

# Seeds déterministes par split
SEED_TRAIN = 10
SEED_VAL   = 20
SEED_TEST  = 30

# ── Split train / val / test ───────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = 1 - TRAIN_RATIO - VAL_RATIO

# ── Entraînement ──────────────────────────────────────────────────────────
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL  = 32
LR               = 1e-3
MAX_EPOCHS       = 30
PATIENCE         = 5             # early stopping

# ── Architecture UNet ─────────────────────────────────────────────────────
UNET_BASE = 64                   # nombre de canaux de base (x2 à chaque niveau)

# ── Architecture GNN ──────────────────────────────────────────────────────
GNN_HIDDEN_DIM  = 128
GNN_NUM_LAYERS  = 4
GNN_KNN_K       = 8              # k voisins pour le graphe de capteurs
GNN_OUTPUT_DIM  = None           # fixé dynamiquement selon le nombre de variables

# ── Loss multivariée ──────────────────────────────────────────────────────
# Poids par variable dans MULTIVAR_KEYS = ["z", "q", "t2m"]
# Augmenter le poids de t2m pour concentrer l'effort sur la température
MULTIVAR_WEIGHTS = [0.5, 0.5, 7.0, 0.5, 0.5]

WEIGHT_UNKNOWN = 0.8             # part de la loss sur les points non observés
WEIGHT_ALL     = 0.2             # part de la loss sur tous les points
LAMBDA_GRAD    = 1.0             # poids du terme gradient (augmenter → moins de lissage)
