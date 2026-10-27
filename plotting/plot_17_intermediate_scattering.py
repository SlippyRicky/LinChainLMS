"""
plot_17_intermediate_scattering.py
==================================
Generates the 2-panel figure illustrating the Self-Intermediate Scattering
Function F_s(q, t) and the Kohlrausch-Williams-Watts (KWW) stretched exponential
relaxation at thermal equilibrium (T = 0.2, q = pi/2).

Inputs:
    - data/intermediate_scattering_N1000.npz

Outputs:
    - report/figures/02_linear_disorder/17_intermediate_scattering.pdf (and .png)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import plot_config
fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
data_dir = plot_config.DATA_DIR

COLORS = {'0.1': '#1f77b4', '0.7': '#ff7f0e', '1.4': '#2ca02c'}

def model_kww(t, A, tau, beta):
    return A * np.exp(-(t / tau)**beta)

def plot_intermediate_scattering():
    print(">>> Generating 17_intermediate_scattering...")
    
    data_path = os.path.join(data_dir, "intermediate_scattering_N1000.npz")
    if not os.path.exists(data_path):
        print(f"[ERROR] Required dataset {data_path} not found. Run simulation/sim_07 first.")
        return
        
    data = np.load(data_path)
    tau_times = data["tau_times"]
    
    disorders = [0.1, 0.7, 1.4]
    
    fig, axes = plt.subplots(1, 2, figsize=(11.76, 4.91))
    ax_a = axes[0]
    ax_b = axes[1]
    
    for eps in disorders:
        k_eps = str(eps)
        c = COLORS[k_eps]
        fs = data[f"fs_{k_eps.replace('.', '_')}"]
        A_fit = float(data[f"A_{eps}"])
        tau_fit = float(data[f"tau_{eps}"])
        beta_fit = float(data[f"beta_{eps}"])
        
        # Subplot (a): F_s(q, t) vs t
        ax_a.plot(tau_times, fs, 'o', color=c, alpha=0.35, markersize=3,
                  label=rf"$\varepsilon = {eps}$ (Simulation)")
        
        t_smooth = np.linspace(0.1, 150.0, 300)
        fit_curve = model_kww(t_smooth, A_fit, tau_fit, beta_fit)
        ax_a.plot(t_smooth, fit_curve, '-', color=c, linewidth=2.0,
                  label=rf"Fit KWW ($\beta = {beta_fit:.2f}$)")
        
        # Subplot (b): -ln(F_s / A) in log-log
        valid = (fs > 0.05) & (tau_times > 0.5) & (fs < A_fit)
        t_val = tau_times[valid]
        y_val = -np.log(np.clip(fs[valid] / A_fit, 1e-6, 1.0))
        
        ax_b.plot(t_val, y_val, 'o', color=c, alpha=0.5, markersize=3.5,
                  label=rf"$\varepsilon = {eps}$ ($\beta = {beta_fit:.2f}$)")
        
        y_fit_val = (t_val / tau_fit)**beta_fit
        ax_b.plot(t_val, y_fit_val, '--', color=c, linewidth=1.5)
        
    ax_a.set_title(r"(a) Relaxation $F_s(q, t)$", fontweight='bold')
    ax_a.set_xlabel(r"Temps $t$ (u.a.)")
    ax_a.set_ylabel(r"$F_s(q, t)$")
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, 1.05)
    ax_a.grid(True, which="both", alpha=0.2, ls='--')
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=12)

    ax_b.set_title(r"(b) Tracé linéarisé KWW", fontweight='bold')
    ax_b.set_xlabel(r"Temps $t$ (u.a.)")
    ax_b.set_ylabel(r"$-\ln(F_s / A)$")
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.grid(True, which="both", alpha=0.2, ls='--')
    ax_b.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=12)

    plt.tight_layout()
    pdf_path = os.path.join(fig_dir, "17_intermediate_scattering.pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("--> 17_intermediate_scattering generated successfully.")

if __name__ == '__main__':
    plot_intermediate_scattering()
