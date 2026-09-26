# Sparse ERA5 Reconstruction and Forecasting

How much of a global weather field can be recovered when only a small fraction of its grid cells are observed? These experiments use simulated sensor locations on the 64 × 32 ERA5 grid from [WeatherBench 2](https://weatherbench2.readthedocs.io/). The reconstruction models receive sparse measurements, a sensor mask, and a nearest-sensor (Voronoi) interpolation.

**Full project report:** [Atmospheric prediction report (PDF, in French)](report/rendu_projet_prevision_atmospherique_CEREA.pdf).

The work covers 2 m temperature alone and multivariate fields (`z`, `q`, `t2m`, `u`, `v`). The default script configuration observes 5% of grid cells. The notebooks also explore other sensor densities and use their own experiment settings, which are recorded inside each notebook.

## Experiments

| Experiment | What it contains |
| --- | --- |
| [Reconstruction](notebooks/reconstruction.ipynb) | StrongUNet, a Voronoi CNN baseline, sensor-density comparisons, and changing training masks |
| [Forecasting](notebooks/forecasting.ipynb) | Reconstruction followed by direct and push-forward forecasts, including rollouts and animated maps |
| [Senseiver](notebooks/senseiver.ipynb) | A self-contained attention-based reconstruction experiment |

The Python scripts provide separate TinyUNet, StrongUNet, and SparseToGridGNN reconstruction pipelines. They share the data loaders, losses, and evaluation functions in this repository. The notebooks contain later, self-contained experiment variants; their settings and saved outputs should be read with each notebook rather than assumed to match the scripts.

## Method and evaluation

The scripts load up to 2,000 six-hourly ERA5 samples from the public WeatherBench 2 Zarr store, split time steps chronologically into 70% training, 15% validation, and 15% test data, and compute normalization statistics on the training split. Sensor masks are simulated with fixed seeds. The multivariate code selects available variables from the dataset.

The scripts report RMSE, RMSE on unobserved cells, and normalized RMSE (RMSE divided by the training-set standard deviation). The U-Net evaluation also reports SSIM. The notebooks include further metrics and plots. Compare results only when the variables, time steps, masks, sensor density, and metric definitions match.

## Run the scripts

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib torch xarray zarr gcsfs fsspec numcodecs
python run_trueunet_univariate.py
```

Other entry points are `run_univariate.py`, `run_multivariate.py`, `run_trueunet_multivariate.py`, and `run_gnn_multivariate.py`. The GNN requires [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) installed for your PyTorch version. The notebooks have additional dependencies and setup cells; the forecasting notebook uses Pillow for GIFs and the plotting cells may use Cartopy.

The ERA5 store requires network access, and the configured time slice needs substantial memory. A GPU is recommended. Change dataset, sensor, and model settings in [`config.py`](config.py). The scripts print metrics and display plots but do not automatically save checkpoints.

## Files

| Path | Purpose |
| --- | --- |
| [`config.py`](config.py) | Data source and experiment settings |
| [`data/`](data/) | ERA5 loading, splits, masks, Voronoi interpolation, and graph datasets |
| [`models/`](models/) | TinyUNet, StrongUNet, and SparseToGridGNN definitions |
| [`train.py`](train.py) | Training losses and U-Net training loops |
| [`evaluate.py`](evaluate.py) | Reconstruction metrics |
| [`visualize.py`](visualize.py) | Field and error plots |
| [`notebooks/`](notebooks/) | Executed reconstruction, forecasting, and Senseiver experiments |

The repository does not include ERA5 data or trained weights. Saved notebook outputs document earlier runs; rerun an experiment to obtain results for your environment.

## References

- [WeatherBench 2 documentation](https://weatherbench2.readthedocs.io/) for the ERA5 dataset.
- [Fukami et al. (2021), *Global field reconstruction from sparse sensors with Voronoi tessellation-assisted deep learning*](https://www.nature.com/articles/s42256-021-00402-2), for the Voronoi-assisted reconstruction approach.
