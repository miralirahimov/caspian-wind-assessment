# Caspian Sea Offshore Wind Assessment

ERA5-based wind resource analysis for the Caspian Sea, covering all five coastal countries (Azerbaijan, Iran, Kazakhstan, Russia, Turkmenistan).

This is the code behind a paper submitted to SOCAR Proceedings (expected 2026).

## What it does

- Loads ERA5 wind data (10 m and 100 m, 2021–2024) from NetCDF files
- Clips grid points to each country's EEZ using Marine Regions v12
- Fits Weibull distributions and calculates Wind Power Density per location
- Identifies top-10% windiest sites (P90 ranking)
- Outputs wind rose and Weibull plots per country, plus summary Excel files

## Stack

Python, xarray, Dask, GeoPandas, Prefect, Cartopy, windrose, SciPy

## Data

NetCDF files aren't included (too large). Get ERA5 wind components (u/v at 10 m and 100 m) from the [Copernicus CDS](https://cds.climate.copernicus.eu/). EEZ file: [Marine Regions v12](https://www.marineregions.org/downloads.php).

Put them in `data/netcdf/` and `data/eez/` — paths are in `config/config.yaml`.

## Usage

```bash
pip install -r requirements.txt
python main.py
```

Plots go to `results/plots/`, summary tables to `results/data/`.
