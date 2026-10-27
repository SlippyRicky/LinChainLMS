"""
plot_15_parameters_comparison.py
================================
Generates the 2-panel comparison of fitted scaling parameters:
  - Subplot A: Fitted scaling exponent alpha (comparing to alpha = 2)
  - Subplot B: Fitted effective scattering strength sigma_eff^2

Inputs:
    - data/mass_disorder.npz
    - data/spring_disorder.npz
    - data/combined_disorder.npz

Outputs:
    - report/figures/02_linear_disorder/15_parameters_comparison.pdf (and .png)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import plot_config
data_dir = plot_config.DATA_DIR

C_TMM = "#1f77b4"       # Blue
C_RICCATI = "#d62728"   # Red
C_PR = "#2ca02c"        # Green
C_ANALYTICAL = "black"

def plot_parameters_comparison():
    print(">>> Generating 15_parameters_comparison...")
    
    disorder_types = ["mass", "spring", "combined"]
    systems = {}
    for dt in disorder_types:
        p = os.path.join(data_dir, f"{dt}_disorder.npz")
        if not os.path.exists(p):
            print(f"[ERROR] Required dataset {p} not found. Run simulation/sim_03 first.")
            return
        systems[dt] = np.load(p)

    methods = ["TMM", "Riccati", "PR", "Analytique"]
    colors = [C_TMM, C_RICCATI, C_PR, C_ANALYTICAL]
    
    fig, (ax_exp, ax_sig) = plt.subplots(1, 2, figsize=(11.76, 4.94))
    x = np.arange(len(disorder_types))  # [0, 1, 2]
    width = 0.18

    # 1. Subplot A: Exponent alpha
    ax_exp.set_title("Exposant de Scaling Ajusté $\\alpha$")
    ax_exp.set_ylabel(r"Exposant $\alpha$")

    for i, method in enumerate(methods):
        if method == "TMM":
            alphas = [float(systems[dt]["tmm_fit_alpha"]) for dt in disorder_types]
        elif method == "Riccati":
            alphas = [float(systems[dt]["riccati_fit_alpha"]) for dt in disorder_types]
        elif method == "PR":
            alphas = [float(systems[dt]["pr_fit_alpha"]) for dt in disorder_types]
        else:
            alphas = [2.0, 2.0, 2.0]
        ax_exp.bar(x + (i - 1.5) * width, alphas, width, label=method, color=colors[i],
                   alpha=0.85, edgecolor="black", linewidth=0.8)

    ax_exp.set_xticks(x)
    ax_exp.set_xticklabels(["Masses", "Ressorts", "Combiné"])
    ax_exp.set_ylim(1.0, 4.2)
    ax_exp.axhline(2.0, color="gray", linestyle="--", linewidth=1.2, label=r"Théorie ($\alpha = 2$)")
    ax_exp.grid(True, which="both", alpha=0.2, ls='--')
    ax_exp.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=13)

    # 2. Subplot B: Effective Scattering Strength sigma_eff^2
    ax_sig.set_title(r"Force du Désordre Effective $\sigma_{\mathrm{eff}}^2$")
    ax_sig.set_ylabel(r"Variance Effective $\sigma_{\mathrm{eff}}^2$")

    eps_val = 0.25
    theo_sigmas = {
        "mass": (eps_val**2) / 3.0,
        "spring": (eps_val**2) / 3.0,
        "combined": 2.0 * (eps_val**2) / 3.0
    }

    for i, method in enumerate(methods):
        if method == "TMM":
            sigmas = [float(systems[dt]["tmm_fit_sigma_eff_sq"]) for dt in disorder_types]
        elif method == "Riccati":
            sigmas = [float(systems[dt]["riccati_fit_sigma_eff_sq"]) for dt in disorder_types]
        elif method == "PR":
            sigmas = [float(systems[dt]["pr_fit_sigma_eff_sq"]) for dt in disorder_types]
        else:
            sigmas = [theo_sigmas[dt] for dt in disorder_types]
        ax_sig.bar(x + (i - 1.5) * width, sigmas, width, label=method, color=colors[i],
                   alpha=0.85, edgecolor="black", linewidth=0.8)

    theo_vals = [theo_sigmas[dt] for dt in disorder_types]
    for pos, val in zip(x, theo_vals):
        ax_sig.plot([pos - 2 * width, pos + 2 * width], [val, val], color="purple", linestyle="-", linewidth=2.5,
                    label=r"Théorie" if pos == 0 else "")

    ax_sig.set_xticks(x)
    ax_sig.set_xticklabels(["Masses", "Ressorts", "Combiné"])
    ax_sig.set_ylim(0.0, 0.06)
    ax_sig.grid(True, which="both", alpha=0.2, ls='--')
    ax_sig.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=13)

    plt.tight_layout()
    fig.savefig("15_parameters_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("--> 15_parameters_comparison generated successfully.")

if __name__ == '__main__':
    plot_parameters_comparison()
