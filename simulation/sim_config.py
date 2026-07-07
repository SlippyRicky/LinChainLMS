import os

# Standalone simulation config to dynamically resolve paths
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SIM_DIR, ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# CENTRALIZED ACCURACY & RESOLUTION SETTINGS
# ==========================================
# Choices:
#   'test'        - low resolution, runs in under 30s to verify code integrity
#   'production'  - high accuracy, publication-grade results (default)
ACCURACY_MODE = 'production'

PARAMS = {
    'test': {
        'sim_01': {'N': 100, 'M': 5, 'n_eps': 10, 'n_omega': 10},
        'sim_02': {'N_diatomic': 500, 'N_asym_dos': 1000},
        'sim_03': {'N_tmm': 5000, 'N_riccati': 200, 'N_green': 100, 'M_tmm': 2},
        'sim_04': {'N_verlet': 1000, 'T_max': 100, 'downsample': 10},
        'sim_05': {'N_verlet': 1000, 'n_freqs': 3},
        'sim_06': {'N_verlet': 2000, 'M_realizations': 2, 'T_max': 1000}
    },
    'production': {
        'sim_01': {'N': 1000, 'M': 100, 'n_eps': 50, 'n_omega': 50},
        'sim_02': {'N_diatomic': 2000, 'N_asym_dos': 4000},
        'sim_03': {'N_tmm': 50000, 'N_riccati': 1000, 'N_green': 300, 'M_tmm': 10},
        'sim_04': {'N_verlet': 6000, 'T_max': 1000, 'downsample': 5},
        'sim_05': {'N_verlet': 6000, 'n_freqs': 15},
        'sim_06': {'N_verlet': 6000, 'M_realizations': 10, 'T_max': 5000}
    }
}

def get_params(script_id):
    """
    Returns a dictionary of parameters corresponding to the chosen ACCURACY_MODE.
    """
    mode = os.environ.get("SIM_ACCURACY_MODE", ACCURACY_MODE)
    if mode not in PARAMS:
        mode = 'production'
    return PARAMS[mode][script_id]
