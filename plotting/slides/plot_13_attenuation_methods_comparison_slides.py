"""
Purpose: Generates the slides-adapted version of the attenuation methods comparison figure.
Inputs: Reads pre-computed datasets (mass_disorder.npz, spring_disorder.npz, combined_disorder.npz, wavepacket_attenuation.npz) from `data/`.
Outputs: Produces 13_attenuation_methods_comparison_slides.pdf and 13_attenuation_methods_comparison_slides.png in `report/figures/slides/`.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Ensure we can import plot_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import plot_config

# --- STYLE CONFIGURATION (Optimized for Presentation Slides) ---
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "legend.fontsize": 12,
    "lines.linewidth": 2.5,  # Thicker lines for readability from afar
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

C_TMM = "#1f77b4"       # Professional Blue
C_RICCATI = "#d62728"   # Professional Red
C_PR = "#2ca02c"        # Professional Green
C_ANALYTICAL = "black"

def plot_slides_attenuation_comparison():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    slides_fig_dir = os.path.join(plot_config.FIGURES_ROOT, 'slides')
    data_dir = plot_config.DATA_DIR

    # Load wavepacket simulation data if available
    wp_data_path = os.path.join(data_dir, "wavepacket_attenuation.npz")
    has_wp_data = os.path.exists(wp_data_path)
    if has_wp_data:
        wp_data = np.load(wp_data_path)
        omega_wp = wp_data["omega"]
        xi_wp_dict = {
            "mass": wp_data["xi_wp_mass"],
            "spring": wp_data["xi_wp_spring"],
            "combined": wp_data["xi_wp_combined"]
        }

    # Setup figure layout (3 rows, 1 column, matching the aspect ratio of plot_08_wavepacket_envelopes_slides)
    fig1, axes1 = plt.subplots(3, 1, figsize=(8.0, 6.5), sharex=True, sharey=True)
    
    sys_keys = ["mass", "spring", "combined"]
    sys_titles = {
        "mass": "Désordre sur les Masses (Diagonal)",
        "spring": "Désordre sur les Ressorts (Hors-Diagonal)",
        "combined": "Désordre Combiné (Masses et Ressorts)"
    }

    for idx, sys_name in enumerate(sys_keys):
        ax = axes1[idx]
        
        # Load the saved NPZ dataset for this disorder type
        npz_path = os.path.join(data_dir, f"{sys_name}_disorder.npz")
        if not os.path.exists(npz_path):
            print(f"Error: Dataset {npz_path} not found. Run the report script to generate it first.")
            return
        
        sys_data = np.load(npz_path)
        omega_vals = sys_data["omega"]
        
        # 1. Plot Participation Ratio
        ax.scatter(sys_data["omega_pr"], sys_data["xi_pr"], color=C_PR, s=8, alpha=0.18,
                   label=r"Taux de Participation $\xi_{PR} = \mathrm{PR}$", zorder=2)
        
        # 2. Plot TMM (Numerical)
        ax.plot(omega_vals, sys_data["xi_tmm"], label="TMM Numérique", lw=2.5, color=C_TMM, zorder=3)
        
        # 3. Plot Riccati (Numerical)
        ax.plot(omega_vals, sys_data["xi_riccati"], label="Riccati Numérique", lw=2.2, linestyle='--', color=C_RICCATI, zorder=4)
        
        # 4. Plot Matsuda-Ishii (Analytical)
        ax.plot(omega_vals, sys_data["xi_analytical"], label="Matsuda-Ishii (Analytique)", lw=1.8, linestyle=':', color=C_ANALYTICAL, zorder=1)
        

        
        # 6. Plot TMM power-law fit curve
        tmm_alpha = float(sys_data["tmm_fit_alpha"])
        tmm_fit_sigma_eff_sq = float(sys_data["tmm_fit_sigma_eff_sq"])
        fit_A = 8.0 / tmm_fit_sigma_eff_sq
        w_fit_curve = np.linspace(0.1, 1.5, 100)
        xi_fit_curve = fit_A * (w_fit_curve ** (-tmm_alpha)) * (1.0 - (w_fit_curve**2) / 4.0)
        
        ax.plot(w_fit_curve, xi_fit_curve, color="darkorange", lw=1.8, linestyle="-.", zorder=5,
                label=f"Ajustement TMM: $\\omega^{{-{tmm_alpha:.2f}}}$")
        
        # 7. Add text box in bottom-left
        pr_alpha = float(sys_data["pr_fit_alpha"])
        text_str = (
            f"$\\alpha_{{TMM}} = {tmm_alpha:.3f}$\n"
            f"$\\alpha_{{PR}} = {pr_alpha:.3f}$\n"
            f"Théorie: $\\alpha = 2.000$"
        )
        ax.text(0.05, 0.05, text_str, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85), zorder=7)
        
        ax.set_title(sys_titles[sys_name], fontweight='bold', fontsize=12, y=1.02)
        
        # Customize y-ticks
        ax.set_yscale('log')
        ax.set_xlim(0, 2.0)
        ax.grid(True, which="both", alpha=0.25, linestyle='--')
        
        if idx == 0:
            ax.legend(bbox_to_anchor=(0.0, 1.35), loc="lower left", frameon=True, facecolor='white', framealpha=0.92, ncol=3, fontsize=8.5, borderaxespad=0.0)
            
        if idx == 2:
            ax.set_xlabel(r"Fréquence $\omega$")
        else:
            ax.tick_params(labelbottom=False)

    # Set y-label only on the middle subplot
    axes1[1].set_ylabel(r"Longueur de localisation $\xi(\omega)$", labelpad=25)
    axes1[0].set_ylim(1.0, 1e5)
    
    plt.tight_layout()
    
    # Save the figures using slide name (handled by custom_figure_savefig/save_figure)
    save_path = os.path.join(slides_fig_dir, "13_attenuation_methods_comparison_slides.pdf")
    save_path_png = os.path.join(slides_fig_dir, "13_attenuation_methods_comparison_slides.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_png, dpi=150, bbox_inches='tight')
    print(f"--> Generated slides-optimized comparison figure: {save_path} and {save_path_png}")
    plt.close(fig1)

if __name__ == '__main__':
    plot_slides_attenuation_comparison()
