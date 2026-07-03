import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import os
from scipy.linalg import eigh
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- STYLE CONFIGURATION (Premium Manuscript) ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 21,
    "legend.fontsize": 17,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.2
})

# Colors
C_ACOUS = "#1f77b4" # Professional Blue
C_OPTIC = "#d62728" # Professional Red
C_GAP   = "#e0e0e0" # Light Gray

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
import plot_config
fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
# Directory creation handled by plot_config
data_dir = plot_config.DATA_DIR




def plot_spectral_asymmetry_dos():
    """Figure 9: Spectral Asymmetry (Mass vs Stiffness DOS + Lifshitz Tails)"""
    data_path = os.path.join(data_dir, "dos_asymmetry_N4000.npz")
    if not os.path.exists(data_path): return
    data = np.load(data_path)
    w_K = data['w_K']
    w_m = data['w_m']
    w_max_perf = 2.0
    
    # 3. Plotting (Side-by-Side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.76, 5.04))
    
    # --- SUBPLOT (a): Linear Spectrum ---
    w_range = np.linspace(0.01, w_max_perf * 0.999, 1000)
    rho_perf = (2 / np.pi) * (1 / np.sqrt(w_max_perf**2 - w_range**2))
    ax1.plot(w_range, rho_perf, color='black', lw=1, linestyle='--', label="Cristal Parfait", alpha=0.6)
    
    bins = 150
    counts_K, b_K = np.histogram(w_K, bins=bins, density=True)
    counts_m, b_m = np.histogram(w_m, bins=bins, density=True)
    
    bc_K = 0.5*(b_K[:-1]+b_K[1:])
    bc_m = 0.5*(b_m[:-1]+b_m[1:])
    
    ax1.fill_between(bc_K, counts_K, color=C_OPTIC, alpha=0.3, label="Désordre de Raideur ($K_n$)")
    ax1.fill_between(bc_m, counts_m, color=C_ACOUS, alpha=0.3, label="Désordre de Masse ($m_n$)")
    
    ax1.set_xlabel(r"Fréquence $\omega$ (u.a.)", fontweight='bold', fontsize=10.5)
    ax1.set_ylabel(r"Densité d'États $\rho(\omega)$", fontweight='bold', fontsize=10.5)
    ax1.set_xlim(0, 2.5)
    ax1.set_ylim(0, 1.5)
    ax1.set_title("(a) Densité d'États Globale", fontsize=10.5)
    ax1.tick_params(axis='both', which='major', labelsize=9)
    ax1.legend(frameon=False, loc='upper left', fontsize=9)
    
    # --- SUBPLOT (b): Logarithmic Lifshitz Tails ---
    ax2.plot(bc_K, counts_K, color=C_OPTIC, lw=2, label="Raideur")
    ax2.plot(bc_m, counts_m, color=C_ACOUS, lw=2, label="Masse")
    ax2.axvline(w_max_perf, color='black', ls='--', lw=1.5, alpha=0.8)
    ax2.text(w_max_perf - 0.05, 1.2, r"$\omega_{max}^0$", ha='right', fontsize=9)
    
    ax2.set_yscale('log')
    # Focus on the tail region: from just before perfect cut-off to the end of disorder spreading
    ax2.set_xlim(1.5, 2.5)
    ax2.set_ylim(1e-2, 1.5)
    ax2.set_xlabel(r"Fréquence $\omega$ (u.a.)", fontweight='bold', fontsize=10.5)
    ax2.set_ylabel(r"$\rho(\omega)$ (échelle log)", fontweight='bold', fontsize=10.5)
    ax2.set_title("(b) Queues de Lifshitz", fontsize=10.5)
    ax2.tick_params(axis='both', which='major', labelsize=9)
    ax2.grid(True, which="both", ls=":", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "05_spectral_asymmetry.pdf"), dpi=300)
    print("--> Reengineered 1x2 DOS: 05_spectral_asymmetry.pdf")



if __name__ == '__main__':
    print('Running plot_spectral_asymmetry_dos...')
    plot_spectral_asymmetry_dos()
    print('Done.')
