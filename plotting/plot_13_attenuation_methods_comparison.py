"""
plot_13_attenuation_methods_comparison.py
=========================================
Generates the 3-panel attenuation comparison figure across the frequency band
for Mass, Spring, and Combined disorder, comparing TMM, Riccati, PR, and theory.

Inputs:
    - data/mass_disorder.npz
    - data/spring_disorder.npz
    - data/combined_disorder.npz
    - data/wavepacket_attenuation.npz (optional overlay)

Outputs:
    - report/figures/02_linear_disorder/13_attenuation_methods_comparison.pdf (and .png)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import plot_config
fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
data_dir = plot_config.DATA_DIR

C_TMM = "#1f77b4"       # Blue
C_RICCATI = "#d62728"   # Red
C_PR = "#2ca02c"        # Green
C_ANALYTICAL = "black"

def plot_attenuation_comparison():
    print(">>> Generating 13_attenuation_methods_comparison...")
    
    sys_keys = ["mass", "spring", "combined"]
    titles = {
        "mass": "Désordre sur les Masses (Diagonal)",
        "spring": "Désordre sur les Ressorts (Hors-Diagonal)",
        "combined": "Désordre Combiné (Masses et Ressorts)"
    }
    
    # Check that datasets exist
    systems_data = {}
    for k in sys_keys:
        p = os.path.join(data_dir, f"{k}_disorder.npz")
        if not os.path.exists(p):
            print(f"[ERROR] Required dataset {p} not found. Run simulation/sim_03 first.")
            return
        systems_data[k] = np.load(p)

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

    fig, axes = plt.subplots(3, 1, figsize=(11.76, 14.69), sharex=True, sharey=True)

    for idx, sys_name in enumerate(sys_keys):
        ax = axes[idx]
        data = systems_data[sys_name]
        omega = data["omega"]
        
        ax.scatter(data["omega_pr"], data["xi_pr"], color=C_PR, s=6, alpha=0.15,
                   label=r"Taux de Participation $\xi_{PR} = \mathrm{PR}$")
        ax.plot(omega, data["xi_tmm"], label="TMM Numérique", lw=2, color=C_TMM)
        ax.plot(omega, data["xi_riccati"], label="Riccati Numérique", lw=1.8, linestyle='--', color=C_RICCATI)
        ax.plot(omega, data["xi_analytical"], label="Matsuda-Ishii (Analytique)", lw=1.5, linestyle=':', color=C_ANALYTICAL)

        if has_wp_data:
            ax.scatter(omega_wp, xi_wp_dict[sys_name], color="darkorange", marker="*", s=100, 
                       edgecolor="black", zorder=5, label="Paquets d'ondes (Simulation)")

        alpha = float(data["tmm_fit_alpha"])
        sigma_sq = float(data["tmm_fit_sigma_eff_sq"])
        w_fit = np.linspace(0.1, 1.5, 100)
        xi_fit = (8.0 / sigma_sq) * (w_fit ** (-alpha)) * (1.0 - (w_fit**2) / 4.0)
        ax.plot(w_fit, xi_fit, color="darkorange", lw=1.5, linestyle="-.",
                label=f"Ajustement TMM: $\\omega^{{-{alpha:.2f}}}$")

        tmm_alpha = float(data["tmm_fit_alpha"])
        pr_alpha = float(data["pr_fit_alpha"])
        text_str = (
            f"$\\alpha_{{TMM}} = {tmm_alpha:.3f}$\n"
            f"$\\alpha_{{PR}} = {pr_alpha:.3f}$\n"
            f"Théorie: $\\alpha = 2.000$"
        )
        ax.text(0.05, 0.05, text_str, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85))

        ax.set_title(titles[sys_name])
        if idx == 2:
            ax.set_xlabel(r"Fréquence $\omega$ (u.a.)")
        ax.set_xlim(0, 2.0)
        ax.set_yscale('log')
        ax.grid(True, which="both", alpha=0.2, ls='--')

        if idx == 0:
            ax.legend(loc='upper right', frameon=True, framealpha=0.9)

    axes[1].set_ylabel(r"Longueur de Localisation $\xi(\omega)$ (u.a.)")
    axes[0].set_ylim(1.0, 1e5)

    plt.tight_layout()
    fig.savefig("13_attenuation_methods_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("--> 13_attenuation_methods_comparison generated successfully.")

if __name__ == '__main__':
    plot_attenuation_comparison()
