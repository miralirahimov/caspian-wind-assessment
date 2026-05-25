import xarray as xr
import pandas as pd
import geopandas as gpd
import yaml
from prefect import task, get_run_logger
import os


@task(name="Load Configuration")
def load_config(config_path: str) -> dict:
    logger = get_run_logger()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return config


@task(name="Load Multi-File NetCDF Dataset")
def load_netcdf(netcdf_path_pattern: str) -> xr.Dataset:
    logger = get_run_logger()
    logger.info(f"Loading NetCDF: {netcdf_path_pattern}")
    try:
        ds = xr.open_mfdataset(netcdf_path_pattern, chunks={'time': 'auto'})
        ds = ds.unify_chunks()

        # ERA5 may use 'valid_time' instead of 'time'
        if 'valid_time' in ds.coords and 'time' not in ds.coords:
            logger.warning("Renaming 'valid_time' to 'time'.")
            ds = ds.rename({'valid_time': 'time'})

        required = ['u10', 'v10', 'u100', 'v100', 'latitude', 'longitude']
        missing = [v for v in required if v not in ds]
        if missing:
            raise ValueError(f"Dataset missing required variables: {missing}")

        logger.info(f"Loaded dataset: {ds}")
        return ds
    except FileNotFoundError:
        logger.error(f"No files found matching: {netcdf_path_pattern}")
        raise
    except Exception as e:
        logger.error(f"Failed to load NetCDF: {e}")
        raise


@task(name="Load EEZ GeoPackage/Shapefile")
def load_eez(eez_path: str) -> gpd.GeoDataFrame:
    logger = get_run_logger()
    if not os.path.exists(eez_path):
        raise FileNotFoundError(f"EEZ file not found: {eez_path}")
    try:
        gdf = gpd.read_file(eez_path)
        logger.info(f"EEZ loaded: {len(gdf)} features, CRS={gdf.crs}")
        return gdf
    except Exception as e:
        logger.error(f"Failed to load EEZ: {e}")
        raise
