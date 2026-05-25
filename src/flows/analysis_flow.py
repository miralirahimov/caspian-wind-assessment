import os
import pandas as pd
import numpy as np
from prefect import flow, get_run_logger
import gc

from src.tasks import data_tasks, geo_tasks, analysis_tasks, plotting_tasks, save_tasks


@flow(name="Caspian Wind Analysis Workflow")
def caspian_wind_analysis_flow(config_path: str = "config/config.yaml"):
    logger = get_run_logger()
    logger.info("Starting Caspian Wind Analysis...")

    config = data_tasks.load_config(config_path=config_path)
    caspian_countries = config.get('caspian_countries', [])
    eez_country_col = config.get('eez_country_column', 'TERRITORY1')
    percentile_val = config.get('percentile_value', 0.90)
    ranking_percentile = config.get('ranking_percentile', 90)
    results_base_path = config.get('results_base_path', 'results')

    plot_path_100m = os.path.join(results_base_path, 'plots', '100m')
    plot_path_10m = os.path.join(results_base_path, 'plots', '10m')
    data_path = os.path.join(results_base_path, 'data')
    for p in [plot_path_100m, plot_path_10m, data_path]:
        os.makedirs(p, exist_ok=True)

    ds = data_tasks.load_netcdf(netcdf_path_pattern=config['netcdf_path_pattern'])
    eez_gdf = data_tasks.load_eez(eez_path=config['eez_file_path'])

    eez_caspian = geo_tasks.filter_and_process_eez(eez_gdf, caspian_countries, eez_country_col)
    territory_mapping_dict, unique_coords_df = geo_tasks.assign_territory_prepare_mapping(ds, eez_caspian, eez_country_col)
    ds = analysis_tasks.calculate_instantaneous_speed_xr(ds=ds, height=100)
    ds = analysis_tasks.calculate_instantaneous_speed_xr(ds=ds, height=10)
    delta_lat, delta_lon = geo_tasks.calculate_grid_resolution_xr(ds=ds)

    plotting_tasks.save_eez_map_task(eez_caspian, os.path.join(results_base_path, 'plots', 'eez_boundaries.png'), config)
    plotting_tasks.save_assignment_map_task(unique_coords_df, territory_mapping_dict, eez_caspian, os.path.join(results_base_path, 'plots', 'territory_assignment.png'), config)

    location_p90_df_100m = analysis_tasks.calculate_pXX_speed_per_location_xr(ds, 100, percentile_val, territory_mapping_dict)
    location_p90_df_10m = analysis_tasks.calculate_pXX_speed_per_location_xr(ds, 10, percentile_val, territory_mapping_dict)

    del ds
    gc.collect()

    final_results_100m = {c: {'area_km2': 0.0, 'coordinates': []} for c in caspian_countries}
    final_geometries_100m = {c: None for c in caspian_countries}
    final_wpd_100m = {c: None for c in caspian_countries}
    final_weibull_100m = {c: {'k': None, 'c': None} for c in caspian_countries}
    final_median_speed_100m = {c: None for c in caspian_countries}

    final_results_10m = {c: {'area_km2': 0.0, 'coordinates': []} for c in caspian_countries}
    final_geometries_10m = {c: None for c in caspian_countries}
    final_wpd_10m = {c: None for c in caspian_countries}
    final_weibull_10m = {c: {'k': None, 'c': None} for c in caspian_countries}
    final_median_speed_10m = {c: None for c in caspian_countries}

    p90_dfs = {100: location_p90_df_100m, 10: location_p90_df_10m}

    for height, results_dict, geom_dict, wpd_dict, weibull_dict, median_dict, plot_path in [
        (100, final_results_100m, final_geometries_100m, final_wpd_100m, final_weibull_100m, final_median_speed_100m, plot_path_100m),
        (10, final_results_10m, final_geometries_10m, final_wpd_10m, final_weibull_10m, final_median_speed_10m, plot_path_10m)
    ]:
        logger.info(f"Processing {height}m...")
        location_pXX_df = p90_dfs[height]
        pXX_col_name = f'p{int(percentile_val*100)}_speed_{height}'

        for country in caspian_countries:
            logger.info(f"{country} — {height}m")
            calculated_area_sq_km = 0.0
            selected_coordinates = []
            clipped_gdf_unprojected = None
            wpd = None
            k, c = None, None
            median_speed = np.nan

            if location_pXX_df is None or 'territory' not in location_pXX_df.columns:
                logger.error(f"P{int(percentile_val*100)} DataFrame missing or lacks 'territory', skipping {country}")
                results_dict[country] = {'area_km2': -1, 'coordinates': []}
                continue

            df_country = location_pXX_df[location_pXX_df['territory'] == country].copy()

            if not df_country.empty and pXX_col_name in df_country.columns and not df_country[pXX_col_name].isnull().all():
                threshold = np.percentile(df_country[pXX_col_name].dropna(), ranking_percentile)
                best_locations_df = df_country[df_country[pXX_col_name] >= threshold].copy()
                num_best = len(best_locations_df)
                logger.info(f"  {num_best} locations above P{ranking_percentile} threshold")

                if num_best > 0 and delta_lat > 0 and delta_lon > 0:
                    selected_coordinates = list(zip(best_locations_df['longitude'], best_locations_df['latitude']))
                    boxes_gdf = analysis_tasks.create_location_boxes(best_locations_df, delta_lat, delta_lon)
                    country_eez_gdf = eez_caspian[eez_caspian[eez_country_col] == country]
                    clipped_gdf_unprojected = geo_tasks.clip_geometries(boxes_gdf, country_eez_gdf)

                    if not clipped_gdf_unprojected.empty:
                        calculated_area_sq_km = analysis_tasks.calculate_clipped_area_km2(clipped_gdf_unprojected, config['caspian_laea_proj4'])
                        geom_dict[country] = clipped_gdf_unprojected

                    if calculated_area_sq_km >= 0 and num_best > 0:
                        ds_reloaded = data_tasks.load_netcdf(config['netcdf_path_pattern'])
                        ds_reloaded = analysis_tasks.calculate_instantaneous_speed_xr(ds_reloaded, height)
                        windiest_ds = analysis_tasks.extract_timeseries_xr(ds_reloaded, selected_coordinates)
                        del ds_reloaded
                        gc.collect()

                        if windiest_ds is not None and 'location' in windiest_ds.dims and windiest_ds.dims['location'] > 0:
                            speed_var = f'instant_speed_{height}'
                            try:
                                speed_series = pd.Series(windiest_ds[speed_var].load().values.ravel()).dropna()
                                if not speed_series.empty:
                                    median_speed = speed_series.median()
                                    logger.info(f"  Median speed: {median_speed:.2f} m/s")
                            except MemoryError:
                                logger.error(f"MemoryError computing median speed for {country} {height}m")
                            except Exception as e:
                                logger.error(f"Median speed failed for {country} {height}m: {e}")

                            wpd = analysis_tasks.calculate_wpd_task_xr(windiest_ds, height)
                            k, c = analysis_tasks.fit_weibull_task_xr(windiest_ds, height)

                            try:
                                windiest_ts_df = windiest_ds.load().to_dataframe().reset_index()
                                plotting_tasks.save_weibull_plot_task(windiest_ts_df, (k, c), country, os.path.join(plot_path, f"{country}_weibull_{height}m.png"), height, config)
                                plotting_tasks.save_wind_rose_plot_task(windiest_ts_df, country, os.path.join(plot_path, f"{country}_windrose_{height}m.png"), height, config)
                                del windiest_ts_df
                            except MemoryError:
                                logger.error(f"MemoryError loading timeseries for plotting {country} {height}m")
                            except Exception as e:
                                logger.error(f"Plotting failed for {country} {height}m: {e}")

                            del windiest_ds
                            gc.collect()
                        else:
                            logger.warning(f"Timeseries empty or extraction failed for {country} {height}m")
            else:
                logger.warning(f"No valid P{int(percentile_val*100)} data for {country} {height}m")

            results_dict[country] = {'area_km2': calculated_area_sq_km, 'coordinates': selected_coordinates}
            if country not in geom_dict:
                geom_dict[country] = None
            wpd_dict[country] = wpd
            weibull_dict[country] = {'k': k, 'c': c}
            median_dict[country] = median_speed

        plotting_tasks.save_windiest_areas_map_task(
            geom_dict, results_dict, eez_caspian,
            os.path.join(plot_path, f"windiest_areas_{height}m.png"), height, config
        )

    logger.info("Aggregating results...")

    def build_summary(results, wpd_d, weibull_d, median_d):
        data = {}
        for c in caspian_countries:
            res = results.get(c, {'area_km2': np.nan, 'coordinates': []})
            wb = weibull_d.get(c, {'k': np.nan, 'c': np.nan})
            area = res['area_km2'] if res['area_km2'] >= 0 else np.nan
            data[c] = {
                'Area (km²)': area,
                'Locations': len(res['coordinates']),
                'Median Speed (m/s)': median_d.get(c, np.nan),
                'WPD (W/m²)': wpd_d.get(c, np.nan),
                'Weibull k': wb.get('k'),
                'Weibull c (m/s)': wb.get('c'),
            }
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index.name = 'Country'
        return df.round({'Area (km²)': 2, 'Median Speed (m/s)': 2, 'WPD (W/m²)': 2, 'Weibull k': 3, 'Weibull c (m/s)': 3})

    summary_100m = build_summary(final_results_100m, final_wpd_100m, final_weibull_100m, final_median_speed_100m)
    summary_10m = build_summary(final_results_10m, final_wpd_10m, final_weibull_10m, final_median_speed_10m)

    save_tasks.save_results_to_excel(output_path=os.path.join(data_path, "caspian_wind_summary_100m.xlsx"), Summary_100m=summary_100m)
    save_tasks.save_results_to_excel(output_path=os.path.join(data_path, "caspian_wind_summary_10m.xlsx"), Summary_10m=summary_10m)
    save_tasks.save_coordinates_to_excel(
        output_path=os.path.join(data_path, "windiest_coordinates.xlsx"),
        results_100m=final_results_100m, results_10m=final_results_10m, countries=caspian_countries
    )

    logger.info("Done. Results saved.")
