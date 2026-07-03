import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 7,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "figure.titlesize": 8,
    "lines.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

script_dir = os.path.dirname(os.path.abspath(__file__))
import plot_config
fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')

def generate_scaling_fits_figure():
    print("\n>>> Generating Scaling Fits Illustration Figure...")
    
    data_dir = plot_config.DATA_DIR
    
    fig, axes = plt.subplots(1, 3, figsize=(6.30, 2.5), sharey=True)
    sys_keys = ["mass", "spring", "combined"]
    titles = ["Désordre de Masses", "Désordre de Ressorts", "Désordre Combiné"]
    
    for idx, sys_name in enumerate(sys_keys):
        ax = axes[idx]
        
        npz_path = os.path.join(data_dir, f"{sys_name}_disorder.npz")
        if not os.path.exists(npz_path):
            print(f"File {npz_path} not found. Skipping {sys_name}.")
            continue
        data = np.load(npz_path)
        omega = data["omega"]
        xi_tmm = data["xi_tmm"]
        
        alpha = data["tmm_fit_alpha"]
        sigma_sq = data["tmm_fit_sigma_eff_sq"]
        
        ax.scatter(omega, xi_tmm, color="royalblue", s=10, marker="x", alpha=0.7, label="Calcul exact TMM")
        
        ax.axvspan(0.4, 1.2, color="gray", alpha=0.15, label="Zone d'ajustement $\omega \in [0.4, 1.2]$")
        
        w_fit_plot = np.linspace(0.1, 1.8, 100)
        xi_fit_plot = (8.0 / sigma_sq) * (w_fit_plot**(-alpha)) * (1.0 - (w_fit_plot**2) / 4.0)
        
        ax.plot(w_fit_plot, xi_fit_plot, color="crimson", lw=2.5, linestyle="--", 
                label=rf"Régression Linéaire ($\alpha = {alpha:.2f}$)")
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(titles[idx])
        ax.set_xlabel(r"Fréquence $\omega$")
        if idx == 0:
            ax.set_ylabel(r"Longueur d'atténuation $\xi$")
            ax.legend(loc="lower left", frameon=True)
            
        ax.grid(True, which="major", ls=":", alpha=0.6)
        
    plt.tight_layout()
    pdf_path = os.path.join(fig_dir, "14_scaling_fits_illustration.pdf")
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    print(f"--> Figure saved to {pdf_path}")

if __name__ == '__main__':
    generate_scaling_fits_figure()
