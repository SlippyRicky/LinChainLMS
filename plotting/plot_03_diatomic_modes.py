import matplotlib.pyplot as plt
from matplotlib.path import Path
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
    "legend.fontsize": 17,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.alpha": 0.2
})

# Colors
C_ACOUS = "#1f77b4" # Professional Blue
C_OPTIC = "#d62728" # Professional Red
C_GAP   = "#e0e0e0" # Light Gray

script_dir = os.path.dirname(os.path.abspath(__file__))
import plot_config
fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
# Directory creation handled by plot_config
data_dir = plot_config.DATA_DIR


def create_centered_bar_circle():
    """Creates a custom Path representing a circle with a mathematically centered diagonal bar."""
    # Circle vertices
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    
    verts = []
    codes = []
    
    # 1. Circle contour
    verts.append((x_circle[0], y_circle[0]))
    codes.append(Path.MOVETO)
    for x, y in zip(x_circle[1:], y_circle[1:]):
        verts.append((x, y))
        codes.append(Path.LINETO)
    verts.append((x_circle[0], y_circle[0]))
    codes.append(Path.CLOSEPOLY)
    
    # 2. Centered diagonal slash / bar (going from bottom-left to top-right inside the circle)
    verts.append((-0.65, -0.65))
    codes.append(Path.MOVETO)
    verts.append((0.65, 0.65))
    codes.append(Path.LINETO)
    
    return Path(verts, codes)


def plot_refined_diatomic():
    """Generates an upgraded premium 3-panel figure representing diatomic physics."""
    modes_path = os.path.join(data_dir, "diatomic_modes.csv")
    if not os.path.exists(modes_path):
        print(f"Error: {modes_path} not found.")
        return

    # Physics Parameters
    m1, m2, K, a = 1.0, 4.0, 1.0, 1.0
    
    # Custom Centered Bar Circle Marker
    custom_marker = create_centered_bar_circle()
    
    # 1. Analytical Dispersion
    qa = np.linspace(-np.pi, np.pi, 500)
    term1 = K * (1/m1 + 1/m2)
    term2 = K * np.sqrt((1/m1 + 1/m2)**2 - (4 * np.sin(qa/2)**2) / (m1*m2))
    omega_plus = np.sqrt(term1 + term2)
    omega_minus = np.sqrt(term1 - term2)
    
    # Setup Figure (1x3 Layout with premium scaling)
    fig, (ax_disp, ax_ac, ax_op) = plt.subplots(1, 3, figsize=(11.76, 4.66))
    
    # --- PANEL A: DISPERSION ---
    ax_disp.plot(qa, omega_plus, color=C_OPTIC, label=r"Branche optique", lw=2)
    ax_disp.plot(qa, omega_minus, color=C_ACOUS, label=r"Branche acoustique", lw=2)
    
    # Gap limits
    gap_min, gap_max = np.sqrt(2*K/m2), np.sqrt(2*K/m1)
    omega_op_max = np.sqrt(2*K*(1/m1 + 1/m2))
    
    ax_disp.axhspan(gap_min, gap_max, color='gray', alpha=0.1)
    ax_disp.text(0, (gap_min+gap_max)/2, r"\textbf{BANDE INTERDITE (GAP)}", ha='center', va='center', fontsize=8, color='gray')
    
    # Vertical grid line boundaries
    ax_disp.axvline(-np.pi, color='gray', linestyle='--', alpha=0.5)
    ax_disp.axvline(np.pi, color='gray', linestyle='--', alpha=0.5)
    
    # Ticks & labels
    ax_disp.set_xlabel(r"Vecteur d'onde $q$", fontweight='bold')
    ax_disp.set_ylabel(r"Pulsation $\omega$", fontweight='bold')
    ax_disp.set_xlim(-np.pi, np.pi)
    ax_disp.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax_disp.set_xticklabels([r'$-\frac{\pi}{a}$', r'$-\frac{\pi}{2a}$', '0', r'$\frac{\pi}{2a}$', r'$\frac{\pi}{a}$'])
    
    ax_disp.set_yticks([0, gap_min, gap_max, omega_op_max])
    ax_disp.set_yticklabels(['0', r'$\omega_{ac}^{\max}$', r'$\omega_{op}^{\min}$', r'$\omega_{op}^{\max}$'])
    
    # Legend Placement (a) - bottom left with clean white frame
    ax_disp.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none', framealpha=0.85)
    ax_disp.set_title(r"\textbf{(a) Relation de dispersion}", fontsize=10, pad=10)

    # --- LOADING MODE DATA FOR PANELS B & C ---
    modes_raw = np.loadtxt(modes_path, delimiter=',')
    sites = modes_raw[:, 0]
    u_acous = modes_raw[:, 1]
    u_optic = modes_raw[:, 2]
    
    idx_m1 = np.arange(0, 20, 2) # Light masses (even sites)
    idx_m2 = np.arange(1, 20, 2) # Heavy masses (odd sites)

    # --- PANEL B: ACOUSTIC MODE ---
    ax_ac.axhline(0, color='black', lw=0.5, alpha=0.5)
    
    # Custom stems: draw lines only, then draw custom scatter points
    markerline_ac, stemlines_ac, baseline_ac = ax_ac.stem(sites, u_acous, linefmt=C_ACOUS, basefmt=" ")
    plt.setp(markerline_ac, visible=False) # Hide uniform markers
    
    # Scatter plot for Light vs Heavy masses (using our custom_marker for perfect centered slash)
    ax_ac.scatter(sites[idx_m1], u_acous[idx_m1], s=55, color='white', edgecolor=C_ACOUS, lw=2, zorder=3, label=r'$m_1$ (atome léger)')
    ax_ac.scatter(sites[idx_m2], u_acous[idx_m2], s=120, color=C_ACOUS, edgecolor='black', marker=custom_marker, lw=1.2, zorder=3, label=r'$m_2$ (atome lourd)')
    
    ax_ac.set_title(r"\textbf{(b) Mode Acoustique ($q \to 0$)}", fontsize=10, pad=10)
    ax_ac.set_xlabel("Site $n$")
    ax_ac.set_ylabel("Déplacement $u_n$")
    ax_ac.set_ylim(-1.3, 1.3)
    
    # Legend Placement (b) - bottom right with clean white frame
    ax_ac.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.85, fontsize=9)

    # --- PANEL C: OPTICAL MODE ---
    ax_op.axhline(0, color='black', lw=0.5, alpha=0.5)
    
    # Custom stems
    markerline_op, stemlines_op, baseline_op = ax_op.stem(sites, u_optic, linefmt=C_OPTIC, basefmt=" ")
    plt.setp(markerline_op, visible=False)
    
    # Scatter plot (using our custom_marker)
    ax_op.scatter(sites[idx_m1], u_optic[idx_m1], s=55, color='white', edgecolor=C_OPTIC, lw=2, zorder=3, label=r'$m_1$ (atome léger)')
    ax_op.scatter(sites[idx_m2], u_optic[idx_m2], s=120, color=C_OPTIC, edgecolor='black', marker=custom_marker, lw=1.2, zorder=3, label=r'$m_2$ (atome lourd)')
    
    ax_op.set_title(r"\textbf{(c) Mode Optique ($q \to 0$)}", fontsize=10, pad=10)
    ax_op.set_xlabel("Site $n$")
    ax_op.set_ylim(-1.3, 1.3)
    
    # Legend Placement (c) - bottom right with clean white frame
    ax_op.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.85, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "03_diatomic_modes.pdf"), dpi=300)
    print("--> Generated upgraded premium 3-panel figure: 03_diatomic_modes.pdf")


if __name__ == '__main__':
    print('Running plot_refined_diatomic...')
    plot_refined_diatomic()
    print('Done.')
