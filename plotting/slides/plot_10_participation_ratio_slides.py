import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import plot_config
from scipy.stats import gaussian_kde

# --- STYLE CONFIGURATION (Optimized for Presentation Slides) ---
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../../"))
slides_fig_dir = os.path.join(plot_config.FIGURES_ROOT, 'slides')
# Directory creation handled by plot_config
data_dir = plot_config.DATA_DIR

def plot_participation_ratio_slides():
    """Generates a slides-optimized 2x3 plot: DOS (top row) over PR (bottom row)"""
    disorders = [0.1, 0.7, 1.4]
    disorder_labels = ["Désordre Faible", "Désordre Intermédiaire", "Désordre Fort"]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]  # Blue, Orange, Red
    
    # 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2), sharex=True)
    
    xlim_max = 2.8
    ylim_dos_max = 1.5
    
    for idx, eps in enumerate(disorders):
        data_path = os.path.join(data_dir, f"disordered_modes_N500_eps{eps}.npz")
        if not os.path.exists(data_path):
            print(f"Data not found for eps={eps}.")
            return
            
        data = np.load(data_path)
        omega = data['omega']
        eigenvectors = data['eigvecs']
        N = len(omega)
        
        # 1. DOS Plot (Top Row, Row 0)
        ax_dos = axes[0, idx]
        kde = gaussian_kde(omega, bw_method=0.08)
        w_smooth = np.linspace(0.001, xlim_max, 300)
        dos = kde(w_smooth)
        
        ax_dos.plot(w_smooth, dos, color=colors[idx], linewidth=2.0)
        ax_dos.fill_between(w_smooth, 0, dos, color=colors[idx], alpha=0.15)
        
        # Cutoff marker (omega_max = 2.0)
        ax_dos.axvline(x=2.0, color="black", linestyle=":", linewidth=1.2, alpha=0.5)
        
        ax_dos.set_ylim(0, ylim_dos_max)
        ax_dos.set_xlim(0, xlim_max)
        ax_dos.set_title(f"{disorder_labels[idx]} ($\sigma_\epsilon = {eps}$)", fontweight='bold', pad=8)
        
        if idx == 0:
            ax_dos.set_ylabel("Densité d'états $g(\omega)$", fontweight="bold")
        ax_dos.grid(True, which="both", linestyle=":", alpha=0.4)
        
        # 2. PR Plot (Bottom Row, Row 1)
        ax_pr = axes[1, idx]
        
        # Calculate PR = 1.0 / sum(eigvecs^4)
        ipr = np.sum(eigenvectors**4, axis=0)
        PR_absolute = 1.0 / ipr
        PR_pct = (PR_absolute / N) * 100.0
        
        # Filter trivial translation modes at very low frequencies
        mask = omega > 0.05
        w_plot = omega[mask]
        PR_plot = PR_pct[mask]
        
        # Draw scatter plot
        ax_pr.scatter(w_plot, PR_plot, color=colors[idx], s=25, alpha=0.7, edgecolor='none', zorder=3)
        
        # Reference limits for PR (in %)
        ax_pr.axhline(y=100.0, color='black', linestyle='--', linewidth=1.2, alpha=0.6, label='Délocalisation (100%)')
        ax_pr.axhline(y=100.0/N, color='red', linestyle='--', linewidth=1.2, alpha=0.6, label=f'Localisation ({100.0/N:.1f}%)')
        ax_pr.axvline(x=2.0, color="black", linestyle=":", linewidth=1.2, alpha=0.5)
        
        ax_pr.set_yscale("log")
        ax_pr.set_ylim(0.7 * 100.0 / N, 150.0)
        ax_pr.set_xlim(0, xlim_max)
        
        # Format y-ticks as percentage
        ticks = [100.0/N, 1.0, 10.0, 100.0]
        tick_labels = [f"{100.0/N:.1f}%", "1%", "10%", "100%"]
        ax_pr.set_yticks(ticks)
        ax_pr.set_yticklabels(tick_labels)
        
        ax_pr.set_xlabel("Fréquence $\omega$")
        
        if idx == 0:
            ax_pr.set_ylabel("Taux de participation $PR (\%)$", fontweight="bold")
            ax_pr.legend(loc="lower left", frameon=True, facecolor='white', framealpha=0.9)
            
        ax_pr.grid(True, which="both", linestyle=":", alpha=0.4)

    plt.tight_layout()
    save_path = os.path.join(slides_fig_dir, "10_participation_ratio_slides.pdf")
    save_path_png = os.path.join(slides_fig_dir, "10_participation_ratio_slides.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_png, dpi=150, bbox_inches='tight')
    print(f"--> Generated slides-optimized DOS over PR figure: {save_path} and {save_path_png}")

if __name__ == '__main__':
    plot_participation_ratio_slides()
