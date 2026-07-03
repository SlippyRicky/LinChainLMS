import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import gaussian_kde

# --- STYLE CONFIGURATION (Premium Manuscript) ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 21,
    "legend.fontsize": 17,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.2
})

script_dir = os.path.dirname(os.path.abspath(__file__))
import plot_config
fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
# Directory creation handled by plot_config
data_dir = plot_config.DATA_DIR

def plot_participation_ratio_spectrum():
    """Figure 11: DOS and Participation Ratio mapping for three disorder levels"""
    disorders = [0.1, 0.7, 1.4]
    disorder_labels = ["faible", "intermédiaire", "fort"]
    # Professional colors for the three levels
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    
    fig, axes = plt.subplots(3, 2, figsize=(11.76, 7.5), sharex=True)
    
    # Common X-axis limit (frequencies go up to ~2.6 for strong disorder)
    xlim_max = 2.8
    ylim_max = 1.5
    
    for idx, eps in enumerate(disorders):
        data_path = os.path.join(data_dir, f"disordered_modes_N500_eps{eps}.npz")
        if not os.path.exists(data_path):
            print(f"Data not found for eps={eps}. Run calc_01_modal_analysis.py first.")
            return
            
        data = np.load(data_path)
        omega = data['omega']
        eigenvectors = data['eigvecs']
        N = len(omega)
        
        # 1. Plot Density of States (DOS) - Left Column
        ax_dos = axes[idx, 0]
        kde = gaussian_kde(omega, bw_method=0.08)
        w_smooth = np.linspace(0.001, xlim_max, 300)
        dos = kde(w_smooth)
        
        ax_dos.plot(w_smooth, dos, color=colors[idx], linewidth=1.8, label=f"$\\sigma_\\epsilon = {eps}$")
        ax_dos.fill_between(w_smooth, 0, dos, color=colors[idx], alpha=0.15)
        
        # Analytical monoatomic cutoff line (omega_max = 2.0)
        ax_dos.axvline(x=2.0, color="black", linestyle=":", linewidth=1.2, alpha=0.5)
        
        ax_dos.set_ylabel(r"$g(\omega)$ (DOS)", fontweight="bold")
        ax_dos.set_ylim(0, ylim_max)
        ax_dos.set_xlim(0, xlim_max)
        ax_dos.set_title(f"Densité d'états — {disorder_labels[idx]} ($\\sigma_\\epsilon = {eps}$)", fontsize=13, loc="left")
        ax_dos.legend(loc="upper left", frameon=False)
        ax_dos.grid(True, which="both", linestyle=":", alpha=0.4)
        
        # 2. Plot Participation Ratio (PR) - Right Column
        ax_pr = axes[idx, 1]
        
        # Calculate PR = 1.0 / sum(eigvecs^4)
        ipr = np.sum(eigenvectors**4, axis=0)
        PR_absolute = 1.0 / ipr
        PR_pct = (PR_absolute / N) * 100.0
        
        # Filter trivial translation modes at very low frequencies
        mask = omega > 0.05
        w_plot = omega[mask]
        PR_plot = PR_pct[mask]
        
        ax_pr.scatter(w_plot, PR_plot, color=colors[idx], s=15, alpha=0.6, edgecolor='none', zorder=3)
        
        # Reference limits for PR (in %)
        ax_pr.axhline(y=100.0, color='black', linestyle='--', linewidth=1.2, alpha=0.6, label=r'Délocalisation ($PR = 100\%$)')
        ax_pr.axhline(y=100.0/N, color='red', linestyle='--', linewidth=1.2, alpha=0.6, label=rf'Localisation ($PR = {100.0/N:.1f}\%$)')
        ax_pr.axvline(x=2.0, color="black", linestyle=":", linewidth=1.2, alpha=0.5)
        
        ax_pr.set_ylabel(r"$PR (\%)$", fontweight="bold")
        ax_pr.set_yscale("log")
        ax_pr.set_ylim(0.7 * 100.0 / N, 150.0)
        
        # Format y-ticks as percentage instead of 10^something
        ticks = [100.0/N, 1.0, 10.0, 100.0]
        tick_labels = [f"{100.0/N:.1f}%", "1%", "10%", "100%"]
        ax_pr.set_yticks(ticks)
        ax_pr.set_yticklabels(tick_labels)
        
        ax_pr.set_title(f"Taux de participation — {disorder_labels[idx]} ($\\sigma_\\epsilon = {eps}$)", fontsize=10, loc="left")
        if idx == 0:
            ax_pr.legend(loc="lower left", frameon=False, ncol=1, fontsize=9)
        ax_pr.grid(True, which="both", linestyle=":", alpha=0.4)

    # Set bottom labels
    axes[2, 0].set_xlabel(r"Fréquence $\omega$ (u.a.)", fontweight="bold")
    axes[2, 1].set_xlabel(r"Fréquence $\omega$ (u.a.)", fontweight="bold")
    
    plt.tight_layout(h_pad=2.0)
    plt.savefig(os.path.join(fig_dir, "10_participation_ratio.pdf"), dpi=300)
    print("--> Generated: 10_participation_ratio.pdf")

if __name__ == '__main__':
    print('Running plot_participation_ratio_spectrum...')
    plot_participation_ratio_spectrum()
    print('Done.')
