import sys
import subprocess
import shutil
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# ==============================================================================
# --- 1. STYLE CONFIGURATION (Premium Manuscript Style) ---
# ==============================================================================
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 21,
    "legend.fontsize": 17,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.2
})

C_TMM = "#1f77b4"       # Professional Blue
C_RICCATI = "#d62728"   # Professional Red
C_PR = "#2ca02c"        # Professional Green
C_ANALYTICAL = "black"

# Directories configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
import plot_config
fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
data_dir = plot_config.DATA_DIR
os.makedirs(data_dir, exist_ok=True)
# Directory creation handled by plot_config

# ==============================================================================
# --- 2. PHYSICS & DYNAMICS ENGINE ---
# ==============================================================================
def compute_linearized_riccati_gamma(N_sites, omega, potential_fluc):
    """
    Vectorized computation of the Linearized Riccati Lyapunov exponent.
    """
    omega_max = 2.0
    a = 1.0
    qa = 2 * np.arcsin(np.clip(omega / omega_max, 0, 0.999))
    q = qa / a
    f_q = 2 * np.exp(-1j * q * a) - 1 - np.exp(-2j * q * a)
    
    A = -f_q * potential_fluc - 0.5 * (f_q**2) * (potential_fluc**2)
    B = np.exp(-2j * q * a) + f_q * np.exp(-2j * q * a) * potential_fluc
    
    B_norm = np.abs(B)
    B = np.where(B_norm > 0, B / B_norm, np.exp(-2j * q * a))
    
    B_cumprod = np.cumprod(B)
    A[0] = 0.0  # Boundary condition: eta_0 = 0
    delta_eta = B_cumprod * np.cumsum(A / B_cumprod)
    
    gamma = np.imag(np.sum(delta_eta)) / N_sites
    return gamma

def run_calculation_and_plot():
    print(">>> Starting physics calculation (N = 6000 sites, Mass vs Spring vs Combined)...")
    
    omega_vals = np.linspace(0.05, 1.95, 200)
    eps_val = 0.25
    N = 6000 
    
    np.random.seed(42)
    # Generate random mass and spring fluctuations
    eps_j_mass = eps_val * (np.random.rand(N) - 0.5) * 2.0
    masses_dis = 1.0 * (1.0 + eps_j_mass)
    
    eps_j_spring = eps_val * (np.random.rand(N + 1) - 0.5) * 2.0
    springs_dis = 1.0 * (1.0 + eps_j_spring)
    
    compliance_fluc = 1.0 / springs_dis - 1.0
    
    # --------------------------------------------------------------------------
    # --- DATA ARRAYS FOR EACH SYSTEM TYPE ---
    # --------------------------------------------------------------------------
    systems = {
        "mass": {
            "title": "Désordre sur les Masses (Diagonal)",
            "masses": masses_dis,
            "springs": np.ones(N + 1),
            "potential_fluc": eps_j_mass,
            "sigma_sq": np.var(eps_j_mass),
            "xi_tmm": np.zeros_like(omega_vals),
            "xi_riccati": np.zeros_like(omega_vals),
            "xi_analytical": np.zeros_like(omega_vals),
            "theo_sigma_eff_sq": np.var(eps_j_mass)
        },
        "spring": {
            "title": "Désordre sur les Ressorts (Hors-Diagonal)",
            "masses": np.ones(N),
            "springs": springs_dis,
            "potential_fluc": compliance_fluc[:N] - np.mean(compliance_fluc[:N]),
            "sigma_sq": np.var(compliance_fluc[:N]),
            "xi_tmm": np.zeros_like(omega_vals),
            "xi_riccati": np.zeros_like(omega_vals),
            "xi_analytical": np.zeros_like(omega_vals),
            "theo_sigma_eff_sq": np.var(compliance_fluc[:N])
        },
        "combined": {
            "title": "Désordre Combiné (Masses et Ressorts)",
            "masses": masses_dis,
            "springs": springs_dis,
            "potential_fluc": (eps_j_mass + compliance_fluc[:N]) - np.mean(eps_j_mass + compliance_fluc[:N]),
            "sigma_sq": np.var(masses_dis / springs_dis[:N] - 1.0),
            "xi_tmm": np.zeros_like(omega_vals),
            "xi_riccati": np.zeros_like(omega_vals),
            "xi_analytical": np.zeros_like(omega_vals),
            "theo_sigma_eff_sq": np.var(eps_j_mass) + np.var(compliance_fluc[:N])
        }
    }
    
    # --------------------------------------------------------------------------
    # --- SPECTRAL CALCULATIONS (TMM, RICCATI, ANALYTICAL) ---
    # --------------------------------------------------------------------------
    for sys_name, sys_data in systems.items():
        print(f"\n   Processing {sys_name} disorder...")
        masses = sys_data["masses"]
        springs = sys_data["springs"]
        potential_fluc = sys_data["potential_fluc"]
        sigma_sq = sys_data["sigma_sq"]
        
        sys_data["xi_analytical"] = (8.0 * (1.0 - (omega_vals**2) / 4.0)) / (sigma_sq * omega_vals**2)
        
        for i, w in enumerate(omega_vals):
            w_sq = w**2
            
            # TMM
            omega_max = 2.0
            a = 1.0
            qa = 2 * np.arcsin(np.clip(w / omega_max, 0, 0.999))
            q = qa / a
            log_g_tmm = 0.0
            r = np.exp(1j * q * a)
            for n in range(N):
                r_next = 1.0 + springs[n]/springs[n+1] - (masses[n] * w_sq)/springs[n+1] - (springs[n]/springs[n+1]) / r
                if np.abs(r_next) < 1e-12: r_next = 1e-12
                log_g_tmm += np.log(np.abs(r_next))
                r = r_next
            gamma_tmm = log_g_tmm / N
            sys_data["xi_tmm"][i] = 1.0 / np.abs(gamma_tmm) if np.abs(gamma_tmm) > 1e-12 else np.nan
            
            # Riccati
            gamma_ric = compute_linearized_riccati_gamma(N, w, potential_fluc)
            sys_data["xi_riccati"][i] = 1.0 / np.abs(gamma_ric) if np.abs(gamma_ric) > 1e-12 else np.nan

    # --------------------------------------------------------------------------
    # --- MODAL PARTICIPATION RATIO (PR) CALCULATIONS ---
    # --------------------------------------------------------------------------
    print("\n>>> Computing Modal Participation Ratio (PR) via tridiagonal diagonalization...")
    N_PR = 6000 
    eps_j_mass_pr = eps_val * (np.random.rand(N_PR) - 0.5) * 2.0
    masses_pr = 1.0 * (1.0 + eps_j_mass_pr)
    
    eps_j_spring_pr = eps_val * (np.random.rand(N_PR + 1) - 0.5) * 2.0
    springs_pr = 1.0 * (1.0 + eps_j_spring_pr)
    
    # 1. Mass Disorder PR
    d_mass = 2.0 / masses_pr
    e_mass = -1.0 / np.sqrt(masses_pr[:-1] * masses_pr[1:])
    eigvals_mass, eigvecs_mass = eigh_tridiagonal(d_mass, e_mass)
    omega_pr_mass = np.sqrt(np.clip(eigvals_mass, 0, None))
    u_mass = eigvecs_mass / np.sqrt(masses_pr)[:, None]
    pr_mass = (np.sum(u_mass**2, axis=0)**2) / np.sum(u_mass**4, axis=0)
    systems["mass"]["omega_pr"] = omega_pr_mass
    systems["mass"]["xi_pr"] = pr_mass
    
    # 2. Spring Disorder PR
    d_spring = springs_pr[:-1] + springs_pr[1:]
    e_spring = -springs_pr[1:-1]
    eigvals_spring, eigvecs_spring = eigh_tridiagonal(d_spring, e_spring)
    omega_pr_spring = np.sqrt(np.clip(eigvals_spring, 0, None))
    u_spring = eigvecs_spring
    pr_spring = (np.sum(u_spring**2, axis=0)**2) / np.sum(u_spring**4, axis=0)
    systems["spring"]["omega_pr"] = omega_pr_spring
    systems["spring"]["xi_pr"] = pr_spring
    
    # 3. Combined Disorder PR
    d_comb = (springs_pr[:-1] + springs_pr[1:]) / masses_pr
    e_comb = -springs_pr[1:-1] / np.sqrt(masses_pr[:-1] * masses_pr[1:])
    eigvals_comb, eigvecs_comb = eigh_tridiagonal(d_comb, e_comb)
    omega_pr_comb = np.sqrt(np.clip(eigvals_comb, 0, None))
    u_comb = eigvecs_comb / np.sqrt(masses_pr)[:, None]
    pr_comb = (np.sum(u_comb**2, axis=0)**2) / np.sum(u_comb**4, axis=0)
    systems["combined"]["omega_pr"] = omega_pr_comb
    systems["combined"]["xi_pr"] = pr_comb

    # --------------------------------------------------------------------------
    # --- POWER-LAW SCALING FITS FOR ALL 12 DATASETS ---
    # --------------------------------------------------------------------------
    print("\n>>> Fitting scaling exponents (omega in [0.4, 1.2], compensating band-edge)...")
    fit_mask = (omega_vals >= 0.4) & (omega_vals <= 1.2)
    w_fit = omega_vals[fit_mask]
    
    for sys_name, sys_data in systems.items():
        sys_data["fits"] = {}
        
        # 1. Fit TMM
        xi_tmm_comp = sys_data["xi_tmm"][fit_mask] / (1.0 - (w_fit**2) / 4.0)
        p = np.polyfit(np.log(w_fit), np.log(xi_tmm_comp), 1)
        sys_data["fits"]["TMM"] = {"alpha": -p[0], "sigma_eff_sq": 8.0 / np.exp(p[1])}
        sys_data["fit_alpha"] = -p[0] # for overlay plotting
        sys_data["fit_A"] = np.exp(p[1])
        
        # 2. Fit Riccati
        xi_ric_comp = sys_data["xi_riccati"][fit_mask] / (1.0 - (w_fit**2) / 4.0)
        p = np.polyfit(np.log(w_fit), np.log(xi_ric_comp), 1)
        sys_data["fits"]["Riccati"] = {"alpha": -p[0], "sigma_eff_sq": 8.0 / np.exp(p[1])}
        
        # 3. Fit Analytical
        xi_ana_comp = sys_data["xi_analytical"][fit_mask] / (1.0 - (w_fit**2) / 4.0)
        p = np.polyfit(np.log(w_fit), np.log(xi_ana_comp), 1)
        sys_data["fits"]["Analytique"] = {"alpha": -p[0], "sigma_eff_sq": 8.0 / np.exp(p[1])}
        
        # 4. Fit PR
        w_pr = sys_data["omega_pr"]
        xi_pr = sys_data["xi_pr"]
        pr_fit_mask = (w_pr >= 0.4) & (w_pr <= 1.2)
        w_pr_fit = w_pr[pr_fit_mask]
        xi_pr_comp = xi_pr[pr_fit_mask] / (1.0 - (w_pr_fit**2) / 4.0)
        p = np.polyfit(np.log(w_pr_fit), np.log(xi_pr_comp), 1)
        sys_data["fits"]["PR"] = {"alpha": -p[0], "sigma_eff_sq": 8.0 / np.exp(p[1])}
        
        print(f"      {sys_name.upper()} DISORDER FITS:")
        for method_name, fit_results in sys_data["fits"].items():
            print(f"         {method_name}: alpha = {fit_results['alpha']:.3f}, sigma_eff_sq = {fit_results['sigma_eff_sq']:.5f}")

    # --------------------------------------------------------------------------
    # --- EXPORT DATA FILES ---
    # --------------------------------------------------------------------------
    print("\n>>> Exporting datasets to att_leng_data/...")
    
    for sys_name, sys_data in systems.items():
        npz_path = os.path.join(data_dir, f"{sys_name}_disorder.npz")
        np.savez(npz_path,
                 omega=omega_vals,
                 xi_tmm=sys_data["xi_tmm"],
                 xi_riccati=sys_data["xi_riccati"],
                 xi_analytical=sys_data["xi_analytical"],
                 omega_pr=sys_data["omega_pr"],
                 xi_pr=sys_data["xi_pr"],
                 tmm_fit_alpha=sys_data["fits"]["TMM"]["alpha"],
                 tmm_fit_sigma_eff_sq=sys_data["fits"]["TMM"]["sigma_eff_sq"],
                 riccati_fit_alpha=sys_data["fits"]["Riccati"]["alpha"],
                 riccati_fit_sigma_eff_sq=sys_data["fits"]["Riccati"]["sigma_eff_sq"],
                 pr_fit_alpha=sys_data["fits"]["PR"]["alpha"],
                 pr_fit_sigma_eff_sq=sys_data["fits"]["PR"]["sigma_eff_sq"])
        
    csv_path = os.path.join(data_dir, "analytical_curves.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["omega", "xi_mass", "xi_spring", "xi_combined"])
        for i in range(len(omega_vals)):
            writer.writerow([
                omega_vals[i],
                systems["mass"]["xi_analytical"][i],
                systems["spring"]["xi_analytical"][i],
                systems["combined"]["xi_analytical"][i]
            ])

    # --------------------------------------------------------------------------
    # --- FIGURE 1: 3-PANEL ATTENUATION CURVES ---
    # --------------------------------------------------------------------------
    print("\n>>> Generating Figure 1: 3-panel attenuation curves...")
    
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
        
    fig1, axes1 = plt.subplots(3, 1, figsize=(11.76, 14.69), sharex=True, sharey=True)
    sys_keys = ["mass", "spring", "combined"]
    
    for idx, sys_name in enumerate(sys_keys):
        ax = axes1[idx]
        sys_data = systems[sys_name]
        
        ax.scatter(sys_data["omega_pr"], sys_data["xi_pr"], color=C_PR, s=6, alpha=0.15,
                   label=r"Taux de Participation $\xi_{PR} = \mathrm{PR}$")
        ax.plot(omega_vals, sys_data["xi_tmm"], label="TMM Numérique", lw=2, color=C_TMM)
        ax.plot(omega_vals, sys_data["xi_riccati"], label="Riccati Numérique", lw=1.8, linestyle='--', color=C_RICCATI)
        ax.plot(omega_vals, sys_data["xi_analytical"], label="Matsuda-Ishii (Analytique)", lw=1.5, linestyle=':', color=C_ANALYTICAL)
        
        if has_wp_data:
            ax.scatter(omega_wp, xi_wp_dict[sys_name], color="darkorange", marker="*", s=100, edgecolor="black", zorder=5, label="Paquets d'ondes (Simulation)")
        
        w_fit_curve = np.linspace(0.1, 1.5, 100)
        xi_fit_curve = sys_data["fit_A"] * (w_fit_curve ** (-sys_data["fit_alpha"])) * (1.0 - (w_fit_curve**2) / 4.0)
        ax.plot(w_fit_curve, xi_fit_curve, color="darkorange", lw=1.5, linestyle="-.",
                label=f"Ajustement TMM: $\\omega^{{-{sys_data['fit_alpha']:.2f}}}$")
        
        tmm_alpha = sys_data["fits"]["TMM"]["alpha"]
        pr_alpha = sys_data["fits"]["PR"]["alpha"]
        text_str = (
            f"$\\alpha_{{TMM}} = {tmm_alpha:.3f}$\n"
            f"$\\alpha_{{PR}} = {pr_alpha:.3f}$\n"
            f"Théorie: $\\alpha = 2.000$"
        )
        ax.text(0.05, 0.05, text_str, transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85))
        
        ax.set_title(sys_data["title"])
        if idx == 2:
            ax.set_xlabel(r"Fréquence $\omega$ (u.a.)")
        ax.set_xlim(0, 2.0)
        ax.set_yscale('log')
        ax.grid(True, which="both", alpha=0.2, ls='--')
        
        # Legend only on the first plot to save space, or inside each.
        # Since it's a full page, inside each is fine but takes space.
        # Let's put legend in the first panel, or simply in each panel if there's room.
        if idx == 0:
            ax.legend(loc='upper right', frameon=True, framealpha=0.9)
            
    # Set y-label on the middle subplot
    axes1[1].set_ylabel(r"Longueur de Localisation $\xi(\omega)$ (u.a.)")
    axes1[0].set_ylim(1.0, 1e5)
    
    plt.tight_layout()
    
    fig1.savefig("13_attenuation_methods_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    print("\n--> Success! All calculations completed, data exported, and figures generated.")

if __name__ == '__main__':
    run_calculation_and_plot()
