import logging
from retirement_simulation import run_model_simulation
from retirement2_helpers import DEFAULTS

# Configure logging at the root level to capture simulator output
logging.basicConfig(level=logging.INFO)

# Run a sample simulation with default settings
# We'll enable debug logging via the settings dictionary
settings = DEFAULTS.copy()
settings["enable_debug_logging"] = True
settings["years"] = 5  # Just a few years for verification

print("Running simulation with logging enabled...")
rows, run_out_age = run_model_simulation(settings)
print(f"\nSimulation complete. Ran for {len(rows)} years.")
