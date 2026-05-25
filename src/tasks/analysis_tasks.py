import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
from scipy.special import gamma
from scipy.stats import weibull_min
from prefect import task, get_run_logger


def _calculate_wind_power_density_weibull(wind_speed_series):
    if not isinstance(wind_speed_series, pd.Series):
        wind_speed_series = pd.Series(wind_speed_series)
    wind_speed_series = wind_speed_series.dropna()
    if wind_speed_series.empty:
        return None
    mean = wind_speed_series.mean()
    std_dev = wind_speed_series.std(ddof=0)
    if mean == 0:
        return 0.0
    if std_dev == 0 and mean > 0:
        return 0.5 * 1.225 * mean**3
    ratio = std_dev / mean
    if ratio <= 0:
        return None
    try:
        k = max(ratio, 1e-6) ** (-1.086)
        k = np.clip(k, 1.0, 10.0)
    except OverflowError:
        k = 10.0
    try:
        c = mean / gamma(1 + (1 / k))
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    try:
        power_density = 0.5 * 1.225 * (c**3) * gamma(1 + (3 / k))
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    if not np.isfinite(power_density) or power_density < 0 or power_density > 50000:
        return None
    return power_density


@task(name="Calculate Instantaneous Speed (Xarray)")
def calculate_instantaneous_speed_xr(ds: xr.Dataset, height: int) -> xr.Dataset:
    logger = get_run_logger()
    u_var, v_var = f'u{height}', f'v{height}'
    speed_var = f'instant_speed_{height}'
    if not all(var in ds for var in [u_var, v_var]):
        raise ValueError(f"Missing '{u_var}' or '{v_var}' in Dataset")
    ds[speed_var] = np.sqrt(ds[u_var]**2 + ds[v_var]**2)
    ds[speed_var].attrs['units'] = ds[u_var].attrs.get('units', 'm s**-1')
    ds[speed_var].attrs['long_name'] = f'{height}m wind speed'
    logger.info(f"Calculated {speed_var}")
    return ds


@task(name="Calculate PXX Speed per Location (Xarray)")
def calculate_pXX_speed_per_location_xr(
    ds: xr.Dataset, height: int, percentile_val: float, territory_map: dict = None
) -> pd.DataFrame:
    logger = get_run_logger()
    speed_var = f'instant_speed_{height}'
    pXX_col_name = f'p{int(percentile_val*100)}_speed_{height}'

    if speed_var not in ds:
        raise ValueError(f"'{speed_var}' not found in Dataset")

    location_pXX_da = ds[speed_var].quantile(percentile_val, dim='time', skipna=True)
    location_pXX_df = location_pXX_da.to_dataframe(name=pXX_col_name).reset_index()

    if not territory_map:
        logger.warning("No territory map provided — returning without territory column")
        return location_pXX_df[[c for c in ['longitude', 'latitude', pXX_col_name] if c in location_pXX_df.columns]]

    try:
        map_keys = list(zip(location_pXX_df['longitude'], location_pXX_df['latitude']))
        location_pXX_df['territory'] = pd.Series(map_keys).map(territory_map)

        valid_count = location_pXX_df['territory'].notna().sum()
        logger.info(f"Territory mapped for {valid_count}/{len(location_pXX_df)} locations")

        if valid_count == 0:
            logger.error("All territory mappings returned NaN — check coordinate key format")
            return location_pXX_df[['longitude', 'latitude', pXX_col_name]]

        location_pXX_df['territory'] = location_pXX_df['territory'].astype('category')
        location_pXX_df = location_pXX_df.dropna(subset=['territory'])

        if location_pXX_df.empty:
            return pd.DataFrame(columns=['longitude', 'latitude', 'territory', pXX_col_name])

        final_cols = ['longitude', 'latitude', 'territory', pXX_col_name]

    except Exception as e:
        logger.error(f"Territory addition failed: {e}", exc_info=True)
        return location_pXX_df[['longitude', 'latitude', pXX_col_name]]

    missing = [c for c in final_cols if c not in location_pXX_df.columns]
    if missing:
        raise ValueError(f"Expected columns missing before return: {missing}")

    return location_pXX_df[final_cols]


@task(name="Create Location Geometry Boxes")
def create_location_boxes(
    selected_locations_df: pd.DataFrame, delta_lat: float, delta_lon: float
) -> gpd.GeoDataFrame:
    logger = get_run_logger()
    if selected_locations_df.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if not {'longitude', 'latitude'}.issubset(selected_locations_df.columns):
        raise ValueError("Missing longitude/latitude columns")
    if delta_lat <= 0 or delta_lon <= 0:
        raise ValueError(f"Invalid grid resolution: dLat={delta_lat}, dLon={delta_lon}")
    geometries = [
        box(
            r['longitude'] - delta_lon / 2, r['latitude'] - delta_lat / 2,
            r['longitude'] + delta_lon / 2, r['latitude'] + delta_lat / 2
        )
        for _, r in selected_locations_df.iterrows()
    ]
    gdf = gpd.GeoDataFrame(selected_locations_df, geometry=geometries, crs="EPSG:4326")
    logger.info(f"Created {len(gdf)} grid boxes")
    return gdf


@task(name="Calculate Clipped Area (km2)")
def calculate_clipped_area_km2(clipped_gdf: gpd.GeoDataFrame, target_crs: str) -> float:
    logger = get_run_logger()
    if clipped_gdf is None or clipped_gdf.empty:
        return 0.0
    try:
        clipped_proj = clipped_gdf.to_crs(target_crs)
        area_km2 = clipped_proj.geometry.area.sum() / 1_000_000
        logger.info(f"Total area: {area_km2:.2f} km²")
        return float(area_km2)
    except Exception as e:
        logger.error(f"Area calculation failed: {e}")
        return -1.0


@task(name="Extract Timeseries for Windiest Locations (Xarray)")
def extract_timeseries_xr(ds: xr.Dataset, target_coords_list: list) -> xr.Dataset | None:
    logger = get_run_logger()
    if not target_coords_list:
        logger.warning("No target coordinates provided")
        return None
    try:
        lats = xr.DataArray([c[1] for c in target_coords_list], dims="location")
        lons = xr.DataArray([c[0] for c in target_coords_list], dims="location")
        windiest_ds = ds.sel(latitude=lats, longitude=lons, method="nearest", drop=False)
        logger.info(f"Extracted timeseries for {len(target_coords_list)} locations")
        return windiest_ds
    except Exception as e:
        logger.error(f"Timeseries extraction failed: {e}")
        return None


@task(name="Calculate WPD for Region (from Xarray)")
def calculate_wpd_task_xr(timeseries_ds: xr.Dataset | None, height: int) -> float | None:
    logger = get_run_logger()
    speed_var = f'instant_speed_{height}'
    if timeseries_ds is None:
        return None
    if speed_var not in timeseries_ds:
        logger.error(f"'{speed_var}' not found in dataset")
        return None
    try:
        speed_series = pd.Series(timeseries_ds[speed_var].load().values.ravel())
        wpd = _calculate_wind_power_density_weibull(speed_series)
        if wpd is not None:
            logger.info(f"WPD ({height}m): {wpd:.2f} W/m²")
        return wpd
    except MemoryError:
        logger.error("MemoryError loading speed data for WPD")
        return None
    except Exception as e:
        logger.error(f"WPD calculation failed: {e}")
        return None


@task(name="Fit Weibull for Region (from Xarray)")
def fit_weibull_task_xr(
    timeseries_ds: xr.Dataset | None, height: int
) -> tuple[float | None, float | None]:
    logger = get_run_logger()
    speed_var = f'instant_speed_{height}'
    if timeseries_ds is None:
        return None, None
    if speed_var not in timeseries_ds:
        logger.error(f"'{speed_var}' not found in dataset")
        return None, None
    try:
        speed_series = pd.Series(timeseries_ds[speed_var].load().values.ravel()).dropna()
        speed_series = speed_series[speed_series > 0.1]
        if speed_series.empty:
            logger.warning("No wind speed data > 0.1 m/s, cannot fit Weibull")
            return None, None
        shape, loc, scale = weibull_min.fit(speed_series, floc=0)
        logger.info(f"Weibull fit ({height}m): k={shape:.3f}, c={scale:.3f}")
        return float(shape), float(scale)
    except MemoryError:
        logger.error("MemoryError loading speed data for Weibull fit")
        return None, None
    except Exception as e:
        logger.error(f"Weibull fit failed: {e}")
        return None, None
