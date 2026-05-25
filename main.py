import os
from src.flows.analysis_flow import caspian_wind_analysis_flow

# Define the base path for results relative to this script's location
# Assumes the default 'results' folder structure defined in config.yaml
RESULTS_BASE = "results"
PLOTS_PATH = os.path.join(RESULTS_BASE, "plots")
DATA_PATH = os.path.join(RESULTS_BASE, "data")
PLOTS_100M_PATH = os.path.join(PLOTS_PATH, "100m")
PLOTS_10M_PATH = os.path.join(PLOTS_PATH, "10m")

def setup_directories():
    """Creates necessary output directories if they don't exist."""
    print("Setting up output directories...")
    os.makedirs(PLOTS_100M_PATH, exist_ok=True)
    os.makedirs(PLOTS_10M_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)
    print("Output directories checked/created.")

if __name__ == "__main__":
    print("--- Starting Caspian Wind Analysis ---")

    # 1. Ensure output directories exist
    setup_directories()

    # 2. Execute the main Prefect analysis flow
    # It will use the default config path "config/config.yaml"
    # defined within the flow function itself.
    # To specify a different config: caspian_wind_analysis_flow(config_path="path/to/other/config.yaml")
    print("\nRunning Prefect flow...")
    state = caspian_wind_analysis_flow() # Execute the flow

    # 3. Print completion message (Prefect also provides detailed logs)
    print("\n--- Caspian Wind Analysis Flow Execution Complete ---")
    # You can check the Prefect logs or UI (if configured) for task details.
    # Results (plots and Excel files) should be in the 'results' directory.