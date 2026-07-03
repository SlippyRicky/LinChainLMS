import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import seaborn as sns
import os

# --- Aesthetic Config ---
sns.set_theme(style="white")
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "figure.dpi": 300
})

script_dir = os.path.dirname(os.path.abspath(__file__))
import plot_config
data_dir = plot_config.DATA_DIR
fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
# Directory creation handled by plot_config

def plot_phase_diagram():
    # Load Data
    path_m = os.path.join(data_dir, "mass_disorder_audit_data.npz")
    path_s = os.path.join(data_dir, "stiffness_disorder_audit_data.npz")
    
    if not os.path.exists(path_m) or not os.path.exists(path_s):
        print("[ERROR] Audit data not found. Run the calculation scripts first.")
        return

    data_m = np.load(path_m)
    data_s = np.load(path_s)
    
    eps_vals = data_m['eps_vals']
    omega_vals = data_m['omega_vals']
    m_mean = data_m['mean_gamma']
    s_mean = data_s['mean_gamma']
    s_std = data_s['std_gamma']

    fig, axes = plt.subplots(2, 2, figsize=(11.76, 8.81))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    ax_flat = axes.flatten()

    # 1. Mass Disorder Phase Map (Foundation)
    im1 = ax_flat[0].pcolormesh(omega_vals, eps_vals, m_mean, shading='gouraud', cmap='magma', 
                             norm=colors.LogNorm(vmin=1e-3, vmax=2))
    ax_flat[0].set_title("a. Analyse Modale : Phase de Localisation", fontweight='bold')
    ax_flat[0].set_ylabel("Amplitude du désordre $\epsilon$")
    ax_flat[0].set_xlabel("Pulsation $\omega$")
    fig.colorbar(im1, ax=ax_flat[0], label=r"Exposant de Lyapunov $\langle \gamma \rangle$")

    # 2. Panel B: Transport Arrest (Dynamics)
    wp_path = os.path.join(data_dir, "wavepacket_arrest_N400.npz")
    if os.path.exists(wp_path):
        wp_data = np.load(wp_path)
        times = wp_data['times']
        data = wp_data['data']
        nodes = np.arange(data.shape[1])
        
        # Calculate width sigma^2(t)
        widths = []
        for t_idx in range(len(times)):
            energy = data[t_idx, :]
            energy = energy / np.sum(energy)
            mean_n = np.sum(nodes * energy)
            width2 = np.sum((nodes - mean_n)**2 * energy)
            widths.append(width2)
        
        ax_flat[1].plot(times, widths, 'firebrick', linewidth=2, label="Désordre ($\epsilon=1.0$)")
        # Add a theoretical diffusive line for reference (slope matches the initial linear/diffusive growth)
        ax_flat[1].plot(times, widths[0] + 11.0 * times, 'k--', alpha=0.5, label="Régime Diffusif")
        ax_flat[1].set_ylim(-200, 4600)
        
        ax_flat[1].set_title("b. Analyse de Transport : Arrêt du Paquet d'Ondes", fontweight='bold')
        ax_flat[1].set_xlabel("Temps $t$")
        ax_flat[1].set_ylabel(r"Largeur du paquet $\sigma^2(t)$")
        ax_flat[1].legend(loc='upper left')
        ax_flat[1].grid(True, alpha=0.3)
        ax_flat[1].text(0.05, 0.55, "Saturation = Localisation", transform=ax_flat[1].transAxes, 
                       color='firebrick', fontweight='bold',
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.15'))

    # 3. Differential Analysis (The Contrast)
    delta_gamma = s_mean - m_mean
    v_ext = np.max(np.abs(delta_gamma)) * 0.7
    im3 = ax_flat[2].pcolormesh(omega_vals, eps_vals, delta_gamma, shading='gouraud', 
                               cmap='RdBu_r', vmin=-v_ext, vmax=v_ext)
    ax_flat[2].set_title(r"c. Contraste ($\gamma_{stiff} - \gamma_{mass}$)", fontweight='bold')
    ax_flat[2].set_ylabel("Amplitude du désordre $\epsilon$")
    ax_flat[2].set_xlabel("Pulsation $\omega$")
    fig.colorbar(im3, ax=ax_flat[2], label="Différence absolue")

    # 4. Statistical Sensitivity
    mask = s_mean > 1e-3
    rel_std = np.zeros_like(s_mean)
    rel_std[mask] = s_std[mask] / s_mean[mask]
    
    im4 = ax_flat[3].pcolormesh(omega_vals, eps_vals, rel_std, shading='gouraud', cmap='viridis',
                             vmin=0, vmax=1.5)
    ax_flat[3].set_title(r"d. Sensibilité Statistique ($\sigma_{\gamma}/\langle \gamma \rangle$)", fontweight='bold')
    ax_flat[3].set_xlabel("Pulsation $\omega$")
    fig.colorbar(im4, ax=ax_flat[3], label="Variance Relative")

    save_path = os.path.join(fig_dir, "16_localization_phase_diagram.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    png_path = os.path.join(fig_dir, "16_localization_phase_diagram.png")
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"--> Generated: {save_path} and {png_path}")

if __name__ == "__main__":
    plot_phase_diagram()
