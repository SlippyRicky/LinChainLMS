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




def plot_eigenmodes_waterfall():
    """Figure 13: Waterfall plot of eigenmodes for strong disorder"""
    # Load decoupled data
    data_path = os.path.join(data_dir, "disordered_modes_N100_eps0.9.npz")
    if not os.path.exists(data_path):
        print("Data not found. Run calc_02_disordered_modes.py first.")
        return
        
    data = np.load(data_path)
    omega = data['omega']
    eigvecs = data['eigvecs']
    N = len(omega)
    
    fig, ax1 = plt.subplots(figsize=(11.76, 6.54))
    
    cmap = plt.cm.turbo
    norm = plt.Normalize(0, np.max(omega))
    
    v_scale = 0.4 # Reduced to avoid excessive overlap while showing mode profile
    # Selection of 15 modes across the spectrum
    n_plot = 15
    indices = np.linspace(0, N - 1, n_plot, dtype=int)
    
    for i in indices:
        u = eigvecs[:, i]
        max_u = np.max(np.abs(u))
        if max_u > 0:
            u = (u / max_u) * v_scale
            
        y_center = omega[i]
        color = cmap(norm(y_center))
        
        is_extreme = (i == indices[0] or i == indices[-1])
        lw = 1.2 if is_extreme else 0.8
        alpha = 1.0 if is_extreme else 0.7
        
        ax1.plot(np.arange(N), y_center + u, color=color, linewidth=lw, alpha=alpha, zorder=3)
        ax1.axhline(y_center, color='gray', linewidth=0.5, alpha=0.15, zorder=1)

    ax1.set_xlabel('Indice du nœud $n$', fontweight='bold')
    ax1.set_ylabel(r'Fréquence $\omega$ (u.a.)', fontweight='bold')
    ax1.set_xlim(0, N-1)
    ax1.set_ylim(0, np.max(omega) + v_scale)
    # ax1.set_title("Anatomie de la localisation spatiale des modes propres ($N=100$, $\\sigma_\\epsilon=0.9$)", fontsize=13, pad=12)
    
    # Add a premium colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax1, orientation='vertical', pad=0.02)
    cbar.set_label(r'Fréquence propre $\omega$ (u.a.)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "07_eigenmodes_waterfall.pdf"), dpi=300)
    print("--> Generated: 07_eigenmodes_waterfall.pdf")



if __name__ == '__main__':
    print('Running plot_eigenmodes_waterfall...')
    plot_eigenmodes_waterfall()
    print('Done.')
