import matplotlib.pyplot as plt
import numpy as np
import os

# --- STYLE CONFIGURATION (Premium Manuscript Style) ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 21,
    "legend.fontsize": 15,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.2
})

# Curated, Harmonious Colors
C_DISP = "#1f77b4" # Professional Blue for dispersion
C_DOS  = "#00897b" # Professional Dark Teal for Density of States
C_GAP  = "#e0e0e0" # Light Gray
C_BG   = "#f5f5f5"

script_dir = os.path.dirname(os.path.abspath(__file__))
import plot_config
fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
# Directory creation handled by plot_config


def plot_dispersion_and_dos():
    """Generates a premium 2-panel figure showing the dispersion relation and density of states with distinct colors and non-overlapping layouts."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, 
        figsize=(11.76, 5.88), 
        sharey=True, 
        gridspec_kw={'width_ratios': [2, 1]}
    )
    
    # --- PANEL A: DISPERSION RELATION ---
    q = np.linspace(-np.pi, np.pi, 500)
    omega = 2 * np.abs(np.sin(q / 2)) # omega_max = 2 (for K/m=1, a=1)
    
    # 1st Brillouin Zone shade
    ax1.axvspan(-np.pi, np.pi, color=C_GAP, alpha=0.08, label=r"1\textsuperscript{ère} Zone de Brillouin")
    
    ax1.plot(q, omega, color=C_DISP, lw=2.2, label=r"Relation de dispersion $\omega(q)$")
    
    ax1.axvline(-np.pi, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(np.pi, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(2, color='gray', linestyle=':', alpha=0.5)
    
    ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax1.set_xticklabels([r'$-\frac{\pi}{a}$', r'$-\frac{\pi}{2a}$', '0', r'$\frac{\pi}{2a}$', r'$\frac{\pi}{a}$'])
    ax1.set_yticks([0, 2])
    ax1.set_yticklabels(['0', r'$\omega_{\max}$'])
    
    ax1.set_xlabel(r"Vecteur d'onde $q$", fontweight='bold')
    ax1.set_ylabel(r"Pulsation $\omega$", fontweight='bold')
    
    # Place legend in bottom-left corner with semi-transparent background to completely avoid data overlap
    ax1.legend(frameon=True, loc='upper center', facecolor='white', edgecolor='none', framealpha=0.85, shadow=False)
    ax1.set_title(r"\textbf{(a) Relation de dispersion}", fontsize=10, pad=10)
    
    # --- PANEL B: DENSITY OF STATES (DOS) ---
    # g(omega) = (2/pi) * (1 / sqrt(omega_max^2 - omega^2))
    # We clip near omega = 2 to avoid division by zero
    omega_vals = np.linspace(0, 1.995, 500)
    dos = (2 / np.pi) * (1 / np.sqrt(4 - omega_vals**2))
    
    ax2.plot(dos, omega_vals, color=C_DOS, lw=2.2, label=r"Densité d'états $g(\omega)$")
    ax2.fill_betweenx(omega_vals, 0, dos, color=C_DOS, alpha=0.1)
    
    # Singularité de Van Hove Horizontal indicator
    ax2.axhline(2, color='gray', linestyle=':', alpha=0.5)
    
    # Labeling and styling Panel B
    ax2.set_xlabel(r"Densité d'états $g(\omega)$", fontweight='bold')
    ax2.set_xlim(0, 2.0)
    ax2.set_xticks([0, 0.318, 1.0, 2.0])
    ax2.set_xticklabels(['0', r'$\frac{2}{\pi\omega_{\max}}$', '1.0', r'$\infty$'])
    
    # Annotating the Van Hove Singularity:
    # Placed more to the right at g(omega) = 1.25 and centered horizontally & vertically
    ax2.annotate(
        r"\textbf{Singularité de}\\\textbf{Van Hove}", 
        xy=(1.92, 1.96), 
        xytext=(1.15, 1.15),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.5, shrinkA=5, shrinkB=5),
        fontsize=9,
        color='black',
        ha='center',
        va='center'
    )
    
    ax2.set_title(r"\textbf{(b) Densité d'états (DOS)}", fontsize=10, pad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "02_dispersion.pdf"), dpi=300)
    print("--> Generated upgraded premium 2-panel figure: 02_dispersion.pdf")


if __name__ == '__main__':
    print('Running plot_dispersion_and_dos...')
    plot_dispersion_and_dos()
    print('Done.')
