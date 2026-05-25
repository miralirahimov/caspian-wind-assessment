import pandas as pd
import os
from prefect import task, get_run_logger


@task(name="Save Summary Results to Excel")
def save_results_to_excel(output_path: str, **dataframes):
    logger = get_run_logger()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, df in dataframes.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    logger.warning(f"Skipping sheet '{sheet_name}' — empty or not a DataFrame")
                    continue
                write_index = df.index.name is not None
                df.to_excel(writer, sheet_name=sheet_name, index=write_index)
                logger.info(f"Saved sheet '{sheet_name}'")
        logger.info(f"Excel saved: {output_path}")
    except ModuleNotFoundError:
        logger.error("openpyxl not found — install it with: pip install openpyxl")
        raise
    except Exception as e:
        logger.error(f"Failed to save Excel {output_path}: {e}")
        raise


@task(name="Save Coordinates to Excel")
def save_coordinates_to_excel(output_path: str, results_100m: dict, results_10m: dict, countries: list):
    logger = get_run_logger()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for height, results in [(100, results_100m), (10, results_10m)]:
                if results is None:
                    continue
                for country in countries:
                    coords = results.get(country, {}).get('coordinates', [])
                    sheet_name = f"{country}_{height}m"[:31]
                    if not coords:
                        continue
                    df = pd.DataFrame(coords, columns=['Longitude', 'Latitude'])
                    df['Longitude'] = df['Longitude'].map('{:.4f}'.format)
                    df['Latitude'] = df['Latitude'].map('{:.4f}'.format)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    logger.info(f"Saved {len(df)} coords to sheet '{sheet_name}'")
        logger.info(f"Coordinates saved: {output_path}")
    except ModuleNotFoundError:
        logger.error("openpyxl not found — install it with: pip install openpyxl")
        raise
    except Exception as e:
        logger.error(f"Failed to save coordinates Excel {output_path}: {e}")
        raise
