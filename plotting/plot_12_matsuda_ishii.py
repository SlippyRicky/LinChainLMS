import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.2
})

C_ACOUS = "#1f77b4"  # Professional Blue for TMM simulation
C_THEORY = "#d62728" # Professional Red for theory

# Directories configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
import plot_config
fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
data_dir = plot_config.DATA_DIR

# Load data
data_path = os.path.join(data_dir, "matsuda_ishii_scaling_multi.npz")
data = np.load(data_path)
omega = data['omega']
eps_values = data['eps_values']

# Create Figure (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(11.76, 4.66))

for i, eps in enumerate(eps_values):
    ax = axes[i]
    gamma_theory = data[f'gamma_theory_{eps}']
    gamma_sim = data[f'gamma_sim_{eps}']
    
    # Calculate localization length xi = 1/gamma
    xi_sim = 1.0 / np.abs(gamma_sim)
    xi_theory = 1.0 / np.abs(gamma_theory)
    
    # Filter out values where gamma is negative or extremely close to 0
    valid = (gamma_sim > 1e-12) & (gamma_theory > 1e-12)
    w_plot = omega[valid]
    xi_sim_plot = xi_sim[valid]
    xi_theory_plot = xi_theory[valid]
    
    # Plot simulated TMM data
    ax.scatter(w_plot, xi_sim_plot, color=C_ACOUS, s=50, marker="+", 
               label="Simulation TMM exacte")
    
    # Plot analytical Matsuda-Ishii theory
    ax.plot(w_plot, xi_theory_plot, color=C_THEORY, lw=2, linestyle="--", 
            label="Théorie de Matsuda-Ishii")
    
    # Format plot
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Fréquence $\omega$")
    if i == 0:
        ax.set_ylabel(r"Longueur de localisation $\xi(\omega)$")
    
    # Custom titles showing the regime
    if eps == 0.1:
        title = r"Faible désordre ($\varepsilon = 0.1$)"
    elif eps == 0.3:
        title = r"Désordre modéré ($\varepsilon = 0.3$)"
    else:
        title = r"Désordre fort ($\varepsilon = 1.4$)"
    ax.set_title(title)
    
    ax.set_xlim(min(w_plot)*0.8, max(w_plot)*1.2)
    
    # Set y-limits to encapsulate both datasets beautifully
    all_xi = np.concatenate([xi_sim_plot, xi_theory_plot])
    ax.set_ylim(min(all_xi)*0.5, max(all_xi)*5.0)
    
    ax.grid(True, which="both", alpha=0.2, ls="--")
    if i == 0:
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)

plt.tight_layout()

# Save the figure as PDF
output_path = os.path.join(fig_dir, "12_matsuda_ishii.pdf")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"--> Saved: {output_path}")
