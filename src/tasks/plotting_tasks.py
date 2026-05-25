import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import weibull_min
from windrose import WindroseAxes
from prefect import task, get_run_logger
import os


def _save_and_close(fig, output_path, dpi=150):
    logger = get_run_logger()
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save {output_path}: {e}")
    finally:
        plt.close(fig)


@task(name="Save EEZ Map Plot")
def save_eez_map_task(eez_caspian: gpd.GeoDataFrame, output_path: str, config: dict):
    logger = get_run_logger()
    if eez_caspian is None or eez_caspian.empty:
        logger.warning("EEZ GeoDataFrame is empty, skipping plot")
        return

    country_col = config.get('eez_country_column', 'TERRITORY1')
    map_extent = config.get('map_extent', (45, 57, 34, 49))
    dpi = config.get('plot_dpi', 150)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    try:
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        ax.add_feature(cfeature.COASTLINE, zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=1)
        eez_caspian.plot(
            ax=ax, column=country_col, edgecolor='black', alpha=0.5, legend=True,
            legend_kwds={'title': "Country", 'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
            zorder=2
        )
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax.set_title("Caspian Sea EEZ Boundaries")
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        _save_and_close(fig, output_path, dpi)
    except Exception as e:
        logger.error(f"EEZ map failed: {e}")
        plt.close(fig)


@task(name="Save Territory Assignment Map Plot")
def save_assignment_map_task(
    unique_coords_df: pd.DataFrame,
    territory_map: dict,
    eez_caspian: gpd.GeoDataFrame,
    output_path: str,
    config: dict
):
    logger = get_run_logger()
    if unique_coords_df is None or unique_coords_df.empty:
        logger.warning("No coordinates to plot, skipping assignment map")
        return
    if territory_map is None:
        logger.warning("No territory map, skipping assignment map")
        return

    map_extent = config.get('map_extent', (45, 57, 34, 49))
    dpi = config.get('plot_dpi', 150)
    plot_eez = eez_caspian is not None and not eez_caspian.empty

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    try:
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        ax.add_feature(cfeature.COASTLINE, zorder=1)

        if plot_eez:
            eez_caspian.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=1, zorder=2)

        if not {'longitude', 'latitude'}.issubset(unique_coords_df.columns):
            logger.error("Missing longitude/latitude in coords DataFrame")
            plt.close(fig)
            return

        assigned, unassigned = [], []
        for _, row in unique_coords_df.iterrows():
            lon, lat = row['longitude'], row['latitude']
            if pd.notna(territory_map.get((lon, lat))):
                assigned.append({'longitude': lon, 'latitude': lat})
            else:
                unassigned.append({'longitude': lon, 'latitude': lat})

        assigned_df = pd.DataFrame(assigned)
        unassigned_df = pd.DataFrame(unassigned)
        logger.info(f"Assigned: {len(assigned_df)}, unassigned: {len(unassigned_df)}")

        if not assigned_df.empty:
            ax.scatter(assigned_df['longitude'], assigned_df['latitude'],
                       color='green', s=5, marker='o', label=f'Assigned ({len(assigned_df)})',
                       transform=ccrs.Geodetic(), zorder=3)
        if not unassigned_df.empty:
            ax.scatter(unassigned_df['longitude'], unassigned_df['latitude'],
                       color='red', s=5, marker='x', label=f'Unassigned ({len(unassigned_df)})',
                       transform=ccrs.Geodetic(), zorder=3)

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        ax.set_title("Territory Assignment Check")
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), title="Status")
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        _save_and_close(fig, output_path, dpi)
    except Exception as e:
        logger.error(f"Assignment map failed: {e}", exc_info=True)
        plt.close(fig)


@task(name="Save Windiest Areas Map Plot")
def save_windiest_areas_map_task(
    clipped_geometries_dict: dict,
    results_dict: dict,
    eez_caspian: gpd.GeoDataFrame,
    output_path: str,
    height: int,
    config: dict
):
    logger = get_run_logger()
    if not clipped_geometries_dict:
        logger.warning("No geometries to plot, skipping windiest areas map")
        return
    if eez_caspian is None or eez_caspian.empty:
        logger.warning("EEZ GeoDataFrame empty, skipping windiest areas map")
        return

    map_extent = config.get('map_extent', (45, 57, 34, 49))
    countries = config.get('caspian_countries', [])
    ranking_percentile = config.get('ranking_percentile', 90)
    percentile_value = config.get('percentile_value', 0.9)
    dpi = config.get('plot_dpi', 150)
    plot_colors = plt.cm.get_cmap('tab10', len(countries))

    fig, ax = plt.subplots(1, 1, figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    try:
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        ax.add_feature(cfeature.COASTLINE, zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=1)
        eez_caspian.plot(ax=ax, edgecolor='blue', facecolor='none', linewidth=0.8, zorder=2)

        legend_handles = []
        for i, country in enumerate(countries):
            gdf = clipped_geometries_dict.get(country)
            color = plot_colors(i)
            if gdf is not None and not gdf.empty:
                gdf.plot(ax=ax, color=color, alpha=0.7, zorder=3)
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7))

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=4)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        ax.set_title(f"Top {100 - ranking_percentile}% Sites by P{int(percentile_value * 100)} Wind Speed ({height}m)")

        legend_labels = [
            f"{c} ({results_dict.get(c, {}).get('area_km2', 0):.1f} km²)"
            if results_dict.get(c, {}).get('area_km2', 0) >= 0 else f"{c} (Error)"
            for c in countries
        ]
        ax.legend(legend_handles, legend_labels, title="Countries & Area",
                  bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        fig.tight_layout(rect=[0, 0, 0.83, 1])
        _save_and_close(fig, output_path, dpi)
    except Exception as e:
        logger.error(f"Windiest areas map failed ({height}m): {e}")
        plt.close(fig)


@task(name="Save Weibull Plot")
def save_weibull_plot_task(
    timeseries_df: pd.DataFrame,
    weibull_params: tuple | None,
    country: str,
    output_path: str,
    height: int,
    config: dict
):
    logger = get_run_logger()
    speed_col = f'instant_speed_{height}'
    hist_color = config.get('color_100m') if height == 100 else config.get('color_10m')
    dpi = config.get('plot_dpi', 150)
    region = f"{country} Windiest Region"

    if timeseries_df is None or timeseries_df.empty:
        logger.warning(f"No timeseries data for {region}, skipping Weibull plot")
        return

    if speed_col not in timeseries_df.columns:
        u_col, v_col = f'u{height}', f'v{height}'
        if u_col in timeseries_df.columns and v_col in timeseries_df.columns:
            timeseries_df[speed_col] = np.sqrt(timeseries_df[u_col]**2 + timeseries_df[v_col]**2)
        else:
            logger.error(f"Missing speed column for {region}, skipping plot")
            return

    wind_data = timeseries_df[speed_col].dropna()
    wind_data = wind_data[wind_data > 0.1]
    if wind_data.empty:
        logger.warning(f"No wind speed data > 0.1 m/s for {region}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        num_bins = min(max(int(np.sqrt(len(wind_data)) / 2), 20), 60)
        ax.hist(wind_data, bins=num_bins, density=True, alpha=0.7, color=hist_color,
                edgecolor='black', label='Observed')

        if weibull_params and all(p is not None for p in weibull_params):
            shape_k, scale_c = weibull_params
            x = np.linspace(0.01, wind_data.quantile(0.995), 300)
            pdf = weibull_min.pdf(x, shape_k, 0, scale_c)
            ax.plot(x, pdf, 'r-', lw=2.5, label=f'Weibull (k={shape_k:.3f}, c={scale_c:.3f})')
            title = f'Wind Speed Distribution — {region} ({height}m)'
        else:
            title = f'Wind Speed Distribution — {region} ({height}m) [fit failed]'

        ax.set_xlabel(f'Wind Speed (m/s) at {height}m')
        ax.set_ylabel('Probability Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.5)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        _save_and_close(fig, output_path, dpi)
    except Exception as e:
        logger.error(f"Weibull plot failed for {region}: {e}")
        plt.close(fig)


@task(name="Save Wind Rose Plot")
def save_wind_rose_plot_task(
    timeseries_df: pd.DataFrame,
    country: str,
    output_path: str,
    height: int,
    config: dict
):
    logger = get_run_logger()
    u_col, v_col = f'u{height}', f'v{height}'
    speed_col = f'instant_speed_{height}'
    cmap = plt.cm.get_cmap(config.get('cmap_100m') if height == 100 else config.get('cmap_10m'))
    dpi = config.get('plot_dpi', 150)
    region = f"{country} Windiest Region"

    if timeseries_df is None or timeseries_df.empty:
        logger.warning(f"No timeseries data for {region}, skipping wind rose")
        return
    if not all(col in timeseries_df.columns for col in [u_col, v_col]):
        logger.error(f"Missing wind component columns for {region}")
        return
    if speed_col not in timeseries_df.columns:
        timeseries_df[speed_col] = np.sqrt(timeseries_df[u_col]**2 + timeseries_df[v_col]**2)

    df = timeseries_df.copy()
    df['wind_direction'] = (np.degrees(np.arctan2(df[v_col], df[u_col])) + 360) % 360
    df.dropna(subset=['wind_direction', speed_col], inplace=True)

    if df.empty:
        logger.warning(f"No valid direction/speed data for {region}")
        return

    fig = plt.figure(figsize=(9, 9))
    try:
        ax = WindroseAxes(fig, [0.1, 0.1, 0.75, 0.75])
        fig.add_axes(ax)

        max_speed = df[speed_col].quantile(0.99)
        spd_bins = np.linspace(0, max(max_speed, 5), 6)
        ax.bar(df['wind_direction'], df[speed_col], bins=spd_bins, nsector=16,
               normed=True, opening=0.8, edgecolor='white', cmap=cmap)

        ax.set_legend(title='Wind Speed (m/s)', loc='center left', bbox_to_anchor=(1.1, 0.5))
        ax.set_title(f"Wind Rose — {region} ({height}m)", y=1.05, fontsize=14)

        yticks = ax.get_yticks()
        if len(yticks) > 1:
            step = max(round(yticks[1] / 2), 2)
            ax.set_yticks(np.arange(0, yticks.max() + step, step))
            ax.set_yticklabels([f'{y:.0f}%' for y in ax.get_yticks()])

        _save_and_close(fig, output_path, dpi)
    except Exception as e:
        logger.error(f"Wind rose failed for {region}: {e}")
        plt.close(fig)
