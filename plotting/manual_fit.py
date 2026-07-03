import numpy as np
import matplotlib.pyplot as plt
import os

# --- STYLE CONFIGURATION ---
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.2
})

# ==========================================
# 1. CHOOSE WHICH DATA TO LOAD
# ==========================================
EPS = 0.7            # Choose between 0.1, 0.7, 1.4
FREQ_IDX = 2         # Index of the frequency (0 to 4)

# Available frequencies roughly correspond to these indices:
# eps=0.1: [0.16, 0.77, 1.41, 1.85, 1.99]
# eps=0.7: [0.15, 0.75, 1.38, 1.83, 2.07]
# eps=1.4: [0.14, 0.69, 1.29, 1.83, 2.21]

# ==========================================
# 2. SET YOUR MANUAL FIT PARAMETERS
# ==========================================
# Exponential Fit: exp(-gamma_exp * X)
GAMMA_EXP = 1.46e-02

# Stretched Exponential Fit: exp(-gamma_stretch * X**beta)
GAMMA_STRETCH = 1.20e-02
BETA = 1.02

# ==========================================
# SCRIPT LOGIC (Do not modify below unless necessary)
# ==========================================
import plot_config
script_dir = os.path.dirname(os.path.abspath(__file__))

def find_data_file(eps, N=6000):
    data_dir = plot_config.DATA_DIR
    for suffix in ["", "_test"]:
        path = os.path.join(data_dir, f"wp_envelopes_eps={eps}_N={N}{suffix}.csv")
        if os.path.exists(path): return path
    return None

data_path = find_data_file(EPS)
if not data_path:
    print(f"Data file for eps={EPS} not found.")
    exit(1)

data = np.loadtxt(data_path, delimiter=',', skiprows=1, ndmin=2)
omegas = data[:, 0]

actual_idx = [1, 5, 10, 15, 19][FREQ_IDX]
if actual_idx >= data.shape[0]: actual_idx = 0

omega = omegas[actual_idx]
N = 6000
col_start = 1 + N // 2
col_end = col_start + 2900
X = np.arange(0, 2900)

log_env = data[actual_idx, col_start:col_end]
amp = np.exp(log_env)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot Data
label = rf"$\omega = {omega:.2f}$"
ax1.semilogy(X, np.maximum(amp, 1e-12), color='#ff7f0e', label=label)
ax2.loglog(X[1:], np.maximum(amp[1:], 1e-12), color='#ff7f0e', label=label)

# Plot Fits
x_plot = X[X >= 5]
y_exp = np.exp(-GAMMA_EXP * x_plot)
y_stretch = np.exp(-GAMMA_STRETCH * x_plot**BETA)

ax1.semilogy(x_plot, y_exp, color='black', linestyle='--', label=f'Exp ($\\gamma$={GAMMA_EXP:.2e})')
ax2.loglog(x_plot, y_stretch, color='black', linestyle=':', label=f'Stretch ($\\gamma$={GAMMA_STRETCH:.2e}, $\\beta$={BETA:.2f})')

ax1.set_ylim(1e-5, 2.0)
ax1.set_xlim(0, 2900)
ax1.set_title(rf"Semi-log ($\varepsilon={EPS}$)")
ax1.legend()

ax2.set_ylim(1e-5, 2.0)
ax2.set_xlim(1, 2900)
ax2.set_title(rf"Log-log ($\varepsilon={EPS}$)")
ax2.legend()

plt.tight_layout()
plt.show()
