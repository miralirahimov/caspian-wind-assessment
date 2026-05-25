import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
from prefect import task, get_run_logger


@task(name="Filter and Process EEZ Data")
def filter_and_process_eez(
    eez_gdf: gpd.GeoDataFrame, countries: list, country_col: str
) -> gpd.GeoDataFrame:
    logger = get_run_logger()
    if country_col not in eez_gdf.columns:
        raise ValueError(f"Column '{country_col}' not found in EEZ data")

    eez_filtered = eez_gdf[eez_gdf[country_col].isin(countries)].copy()
    eez_filtered = eez_filtered[eez_filtered.is_valid]
    logger.info(f"Filtered to {len(eez_filtered)} valid EEZ features")

    if eez_filtered.empty:
        logger.warning("No valid EEZ geometries found")
        return eez_filtered

    try:
        eez_processed = eez_filtered.dissolve(by=country_col).reset_index()
    except Exception as e:
        logger.error(f"Dissolve failed: {e}. Returning undissolved.")
        eez_processed = eez_filtered.reset_index(drop=True)

    return eez_processed


@task(name="Calculate Grid Resolution (from Xarray)")
def calculate_grid_resolution_xr(ds: xr.Dataset) -> tuple[float, float]:
    logger = get_run_logger()
    delta_lat, delta_lon = 0.0, 0.0
    try:
        unique_lats = np.unique(ds['latitude'].values)
        unique_lons = np.unique(ds['longitude'].values)
        if len(unique_lats) > 1:
            delta_lat = float(np.median(np.diff(unique_lats)))
        if len(unique_lons) > 1:
            delta_lon = float(np.median(np.diff(unique_lons)))
        logger.info(f"Grid resolution: dLat={delta_lat:.4f}°, dLon={delta_lon:.4f}°")
    except Exception as e:
        logger.error(f"Error calculating grid resolution: {e}")
    return delta_lat, delta_lon


@task(name="Assign Territory (Prepare Mapping)")
def assign_territory_prepare_mapping(
    ds: xr.Dataset, eez_caspian: gpd.GeoDataFrame, country_col: str
) -> tuple[dict, pd.DataFrame]:
    logger = get_run_logger()
    if eez_caspian.empty:
        logger.warning("EEZ GeoDataFrame is empty")
        return {}, pd.DataFrame(columns=['longitude', 'latitude'])

    lat_coords = ds['latitude'].load().values
    lon_coords = ds['longitude'].load().values
    multi_index = pd.MultiIndex.from_product([lat_coords, lon_coords], names=['latitude', 'longitude'])
    unique_coords_df = multi_index.to_frame(index=False)[['longitude', 'latitude']]
    logger.info(f"Extracted {len(unique_coords_df)} unique coordinate pairs")

    geometry = gpd.points_from_xy(unique_coords_df['longitude'], unique_coords_df['latitude'])
    unique_points_gdf = gpd.GeoDataFrame(unique_coords_df, geometry=geometry, crs="EPSG:4326")

    if unique_points_gdf.crs != eez_caspian.crs:
        eez_caspian = eez_caspian.to_crs(unique_points_gdf.crs)

    points_with_territory = gpd.sjoin(
        unique_points_gdf, eez_caspian[[country_col, 'geometry']],
        how="left", predicate="within"
    )
    points_with_territory = points_with_territory[~points_with_territory.index.duplicated(keep='first')]

    territory_map_data = points_with_territory[[country_col]].rename(columns={country_col: 'territory'})
    unique_coords_df_with_territory = unique_coords_df.join(territory_map_data, how='left')

    assigned = unique_coords_df_with_territory['territory'].notna().sum()
    logger.info(f"Territory assigned to {assigned}/{len(unique_coords_df)} locations")

    territory_mapping_dict = unique_coords_df_with_territory.set_index(['longitude', 'latitude'])['territory'].to_dict()
    return territory_mapping_dict, unique_coords_df


@task(name="Clip Geometries")
def clip_geometries(gdf_to_clip: gpd.GeoDataFrame, clip_boundary_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger = get_run_logger()
    if gdf_to_clip is None or gdf_to_clip.empty or clip_boundary_gdf is None or clip_boundary_gdf.empty:
        crs = gdf_to_clip.crs if gdf_to_clip is not None else "EPSG:4326"
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    try:
        if gdf_to_clip.crs != clip_boundary_gdf.crs:
            gdf_to_clip = gdf_to_clip.to_crs(clip_boundary_gdf.crs)
        clipped = gpd.clip(gdf_to_clip, clip_boundary_gdf, keep_geom_type=True)
        logger.info(f"Clipped to {len(clipped)} geometries")
        return clipped
    except Exception as e:
        logger.error(f"Clipping failed: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=gdf_to_clip.crs)


@task(name="Project GeoDataFrame")
def project_gdf(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    logger = get_run_logger()
    if gdf is None or gdf.empty:
        return gdf
    try:
        return gdf.to_crs(target_crs)
    except Exception as e:
        logger.error(f"Projection to '{target_crs}' failed: {e}")
        raise
