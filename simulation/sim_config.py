import os

# Standalone simulation config to dynamically resolve paths
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SIM_DIR, ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)
