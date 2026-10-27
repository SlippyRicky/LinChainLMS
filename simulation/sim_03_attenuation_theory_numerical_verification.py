"""
sim_03_attenuation_theory_numerical_verification.py
===================================================
Computes Lyapunov exponents (attenuation/localization lengths) using Transfer Matrix
Method (TMM), Linearized Riccati walks, and Modal Participation Ratio (PR) to verify
perturbation theory (Matsuda-Ishii scaling) for diagonal (mass), off-diagonal (spring),
and combined disorders.

Accelerated with Numba JIT for high-performance execution.

Outputs:
    - data/matsuda_ishii_scaling_multi.npz
    - data/riccati_walk_N1000.npz
    - data/harmonic_envelope_N300.npz
    - data/homogenization_bounds_sweep.csv
    - data/mass_disorder.npz
    - data/spring_disorder.npz
    - data/combined_disorder.npz
    - data/analytical_curves.csv
    - data/attenuation_methods_comparison.npz
"""
import numpy as np
import pandas as pd
import os
import csv
import time
from numba import njit
from scipy.stats import linregress
from scipy.linalg import eigh_tridiagonal

import sim_config
data_dir = sim_config.DATA_DIR

# =================================================================
# Fast JIT-Compiled TMM Kernels
# =================================================================
@njit(fastmath=True)
def _tmm_mono_kernel(omega_vals, m_arr, K, N):
    """JIT accelerated TMM loop for mass disorder."""
    n_w = len(omega_vals)
    gamma_out = np.zeros(n_w)
    for i in range(n_w):
        w = omega_vals[i]
        w_sq = w**2
        arg_asin = w / 2.0
        if arg_asin < 0.0:
            arg_asin = 0.0
        elif arg_asin > 0.999:
            arg_asin = 0.999
        qa = 2.0 * np.arcsin(arg_asin)
        r = np.exp(1j * qa)
        log_g = 0.0
        for n in range(N):
            r_next = (2.0 - m_arr[n] * w_sq / K) - 1.0 / r
            if np.abs(r_next) < 1e-12:
                r_next = 1e-12 + 0j
            log_g += np.log(np.abs(r_next))
            r = r_next
        gamma_out[i] = log_g / N
    return gamma_out


@njit(fastmath=True)
def _tmm_general_kernel(omega_vals, masses, springs, N):
    """JIT accelerated TMM loop for general mass/spring disorder."""
    n_w = len(omega_vals)
    xi_out = np.zeros(n_w)
    for i in range(n_w):
        w = omega_vals[i]
        w_sq = w**2
        arg_asin = w / 2.0
        if arg_asin < 0.0:
            arg_asin = 0.0
        elif arg_asin > 0.999:
            arg_asin = 0.999
        qa = 2.0 * np.arcsin(arg_asin)
        r = np.exp(1j * qa)
        log_g = 0.0
        for n in range(N):
            ratio_k = springs[n] / springs[n+1]
            r_next = (1.0 + ratio_k) - (masses[n] * w_sq) / springs[n+1] - ratio_k / r
            if np.abs(r_next) < 1e-12:
                r_next = 1e-12 + 0j
            log_g += np.log(np.abs(r_next))
            r = r_next
        gamma = log_g / N
        abs_gamma = np.abs(gamma)
        xi_out[i] = 1.0 / abs_gamma if abs_gamma > 1e-12 else 1e12
    return xi_out


def compute_tmm_single_eps(omega_vals, eps, N, M):
    """Computes ensemble-averaged TMM Lyapunov exponents."""
    gamma_sim = np.zeros_like(omega_vals)
    K = 1.0
    m0 = 1.0
    for seed in range(M):
        np.random.seed(seed)
        m_arr = m0 * (1.0 + eps * (np.random.rand(N) - 0.5))
        gamma_sim += _tmm_mono_kernel(omega_vals, m_arr, K, N)
    gamma_sim /= M
    return gamma_sim


def compute_linearized_riccati_gamma(N_sites, omega, potential_fluc):
    """Vectorized computation of the Linearized Riccati Lyapunov exponent."""
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
    A[0] = 0.0
    delta_eta = B_cumprod * np.cumsum(A / B_cumprod)
    
    gamma = np.imag(np.sum(delta_eta)) / N_sites
    return gamma


# =================================================================
# MAIN COMPUTATION BLOCKS
# =================================================================

def compute_matsuda_ishii():
    """Validates Matsuda-Ishii scaling against fast exact TMM."""
    print("\n[Theory] Computing Matsuda-Ishii (TMM Scaling)...")
    t0 = time.time()
    params = sim_config.get_params("sim_03")
    N = params['N_tmm']
    M = params['M_tmm']
    
    omega_vals = np.logspace(-2, 0, 40)
    eps_values = [0.1, 0.3, 1.4]
    results = {}
    
    for eps in eps_values:
        sigma_sq = (eps**2) / 12.0
        gamma_theory = (sigma_sq / 8.0) * omega_vals**2 / (1.0 - omega_vals**2 / 4.0)
        gamma_sim = compute_tmm_single_eps(omega_vals, eps, N, M)
        results[f"gamma_theory_{eps}"] = gamma_theory
        results[f"gamma_sim_{eps}"] = gamma_sim
        print(f"    -> eps={eps} done")
        
    filename = "matsuda_ishii_scaling_multi.npz"
    np.savez_compressed(os.path.join(data_dir, filename), 
             omega=omega_vals, eps_values=np.array(eps_values), **results)
    print(f"  -> Done in {time.time()-t0:.2f}s. Saved: {filename}")


def compute_riccati_walk():
    """Single trajectory of Riccati phase random walk."""
    print("\n[Theory] Computing Riccati Walk...")
    params = sim_config.get_params("sim_03")
    N = params['N_riccati']
    
    np.random.seed(42)
    eps_val = 0.3
    eps = eps_val * (np.random.rand(N) - 0.5) * 2.0
    
    w = 1.0
    a = 1.0
    q0 = 2 * np.arcsin(w / 2.0)
    
    delta_eta = np.zeros(N, dtype=complex)
    R = np.zeros(N, dtype=complex)
    f_q = 2 * np.exp(-1j * q0 * a) - 1 - np.exp(-2j * q0 * a)
    
    for n in range(1, N):
        A_n = -f_q * eps[n] - 0.5 * (f_q**2) * (eps[n]**2)
        B_n = np.exp(-2j * q0 * a) + f_q * np.exp(-2j * q0 * a) * eps[n]
        B_n = B_n / np.abs(B_n) if np.abs(B_n) > 0 else np.exp(-2j * q0 * a)
        
        delta_eta[n] = A_n + B_n * delta_eta[n-1]
        R[n] = np.exp(1j * (q0 * a + delta_eta[n]))
        
    eta_walk = np.cumsum(delta_eta)
    filename = "riccati_walk_N1000.npz"
    np.savez_compressed(os.path.join(data_dir, filename), eta_walk=eta_walk, R=R, q0=q0 * a)
    print(f"  -> Saved {filename}")


def compute_harmonic_envelope():
    """Steady-state envelope of harmonically driven chain."""
    print("\n[Theory] Computing Harmonic Envelope (Green's Function)...")
    t0 = time.time()
    params = sim_config.get_params("sim_03")
    N = params['N_green']
    
    epsilon_max = 0.3
    M0, K0 = 1.0, 1.0
    omega = np.sqrt(2)
    g = 0.01  
    
    np.random.seed(42)
    epsilon = np.random.uniform(-epsilon_max, epsilon_max, N)
    m = M0 * (1 + epsilon)
    
    u = np.zeros(N)
    v = np.zeros(N)
    a = np.zeros(N)
    envelope = np.zeros(N)
    dt = 0.05
    STEPS = 20000
    
    for step in range(STEPS):
        t = step * dt
        u[1:] += v[1:] * dt + 0.5 * a[1:] * dt**2
        u[0] = np.sin(omega * t)
        
        a_old = a.copy()
        forces = K0 * (u[2:] + u[:-2] - 2 * u[1:-1]) - g * v[1:-1]
        a[1:-1] = forces / m[1:-1]
        force_last = K0 * (0.0 + u[-2] - 2 * u[-1]) - g * v[-1]
        a[-1] = force_last / m[-1]
        v[1:] += 0.5 * dt * (a_old[1:] + a[1:])
        
        if step > STEPS - 8000:
            np.maximum(envelope, np.abs(u), out=envelope)

    nodes = np.arange(N)
    fit_mask = (nodes >= 10) & (nodes <= (N // 2)) & (envelope > 1e-6)
    fit_nodes = nodes[fit_mask]
    fit_u = envelope[fit_mask]
    slope, intercept, _, _, _ = linregress(fit_nodes, np.log(fit_u))
    gamma = -slope
    theory_env = np.exp(intercept - gamma * nodes)
    
    filename = "harmonic_envelope_N300.npz"
    np.savez_compressed(os.path.join(data_dir, filename), 
             nodes=nodes, u_max=envelope, 
             node_peaks=nodes, u_peaks=envelope, 
             theory_env=theory_env, gamma=gamma, N=N)
    print(f"  -> Done in {time.time()-t0:.2f}s. Saved: {filename}")


def compute_homogenization_bounds():
    """Effective stiffness bounds (Voigt / Reuss)."""
    print("\n[Theory] Computing Homogenization Bounds...")
    t0 = time.time()
    eps_values = np.linspace(0.01, 1.5, 30)
    voigt, reuss, numerical_single = [], [], []
    K_ref = 1.0 
    N_sites = 6000 
    np.random.seed(42) 
    
    for eps in eps_values:
        v = K_ref
        r = (eps * K_ref) / (np.log(1 + 0.5*eps) - np.log(1 - 0.5*eps))
        voigt.append(v)
        reuss.append(r)
        k_rand = K_ref * (1.0 + eps * (np.random.rand(N_sites) - 0.5))
        k_eff = 1.0 / np.mean(1.0 / k_rand)
        numerical_single.append(k_eff)
        
    df = pd.DataFrame({
        "Epsilon": eps_values, 
        "Voigt": voigt, 
        "Reuss": reuss,
        "Numerical_Mean": numerical_single,
        "Numerical_Std": np.zeros(len(eps_values)),
        "Numerical_Single": numerical_single
    })
    path = os.path.join(data_dir, "homogenization_bounds_sweep.csv")
    df.to_csv(path, index=False)
    print(f"  -> Done in {time.time()-t0:.2f}s. Saved: homogenization_bounds_sweep.csv")


def compute_attenuation_methods():
    """
    Master comparative analysis of spatial attenuation length (xi) across the band
    for Diagonal (mass), Off-Diagonal (spring), and Combined disorders using:
      1. Transfer Matrix Method (TMM) [JIT accelerated]
      2. Linearized Riccati Formalism
      3. Participation Ratio of Exact Eigenmodes (PR)
      4. Matsuda-Ishii Analytical Perturbation Theory

    Saves:
      - mass_disorder.npz
      - spring_disorder.npz
      - combined_disorder.npz
      - analytical_curves.csv
      - attenuation_methods_comparison.npz
    """
    print("\n[Theory] Computing Attenuation Methods Comparison (Mass, Spring, Combined)...")
    t0 = time.time()
    
    mode = os.environ.get("SIM_ACCURACY_MODE", sim_config.ACCURACY_MODE)
    N = 6000 if mode == 'production' else 2000
    omega_vals = np.linspace(0.05, 1.95, 200 if mode == 'production' else 60)
    eps_val = 0.25
    
    np.random.seed(42)
    eps_j_mass = eps_val * (np.random.rand(N) - 0.5) * 2.0
    masses_dis = 1.0 * (1.0 + eps_j_mass)
    
    eps_j_spring = eps_val * (np.random.rand(N + 1) - 0.5) * 2.0
    springs_dis = 1.0 * (1.0 + eps_j_spring)
    
    compliance_fluc = 1.0 / springs_dis - 1.0
    
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
    
    for sys_name, sys_data in systems.items():
        masses = sys_data["masses"]
        springs = sys_data["springs"]
        potential_fluc = sys_data["potential_fluc"]
        sigma_sq = sys_data["sigma_sq"]
        
        sys_data["xi_analytical"] = (8.0 * (1.0 - (omega_vals**2) / 4.0)) / (sigma_sq * omega_vals**2)
        
        # Fast JIT TMM
        sys_data["xi_tmm"] = _tmm_general_kernel(omega_vals, masses, springs, N)
        
        # Vectorized Riccati
        for i, w in enumerate(omega_vals):
            gamma_ric = compute_linearized_riccati_gamma(N, w, potential_fluc)
            sys_data["xi_riccati"][i] = 1.0 / np.abs(gamma_ric) if np.abs(gamma_ric) > 1e-12 else 1e12

    # Modal Participation Ratio (PR)
    N_PR = min(N, 6000)
    for sys_name, sys_data in systems.items():
        m_sys = sys_data["masses"][:N_PR]
        k_sys = sys_data["springs"][:N_PR+1]
        
        d = (k_sys[:-1] + k_sys[1:]) / m_sys
        e = -k_sys[1:-1] / np.sqrt(m_sys[:-1] * m_sys[1:])
        eigvals, eigvecs = eigh_tridiagonal(d, e)
        w_pr = np.sqrt(np.clip(eigvals, 0, None))
        u_sq = eigvecs**2
        pr = (np.sum(u_sq, axis=0)**2) / np.sum(u_sq**2, axis=0)
        sys_data["omega_pr"] = w_pr
        sys_data["xi_pr"] = pr

    # Fit scaling exponents in acoustic window [0.4, 1.2]
    fit_mask = (omega_vals >= 0.4) & (omega_vals <= 1.2)
    w_fit = omega_vals[fit_mask]
    
    for sys_name, sys_data in systems.items():
        sys_data["fits"] = {}
        
        # 1. Fit TMM
        xi_tmm_comp = sys_data["xi_tmm"][fit_mask] / (1.0 - (w_fit**2) / 4.0)
        p = np.polyfit(np.log(w_fit), np.log(xi_tmm_comp), 1)
        sys_data["fits"]["TMM"] = {"alpha": -p[0], "sigma_eff_sq": 8.0 / np.exp(p[1])}
        
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

    # Save individual disorder files (used by plot_14 and plot_15)
    for sys_name, sys_data in systems.items():
        npz_path = os.path.join(data_dir, f"{sys_name}_disorder.npz")
        np.savez_compressed(
            npz_path,
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
            pr_fit_sigma_eff_sq=sys_data["fits"]["PR"]["sigma_eff_sq"]
        )
        print(f"  -> Saved: {sys_name}_disorder.npz")
        
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
    print("  -> Saved: analytical_curves.csv")

    comp_path = os.path.join(data_dir, "attenuation_methods_comparison.npz")
    np.savez_compressed(
        comp_path,
        omega=omega_vals,
        xi_tmm=systems["mass"]["xi_tmm"],
        xi_riccati=systems["mass"]["xi_riccati"],
        xi_analytical=systems["mass"]["xi_analytical"],
        omega_pr=systems["mass"]["omega_pr"],
        xi_pr=systems["mass"]["xi_pr"],
        epsilon=eps_val
    )
    print(f"  -> Done in {time.time()-t0:.2f}s. Saved: attenuation_methods_comparison.npz")


if __name__ == '__main__':
    print("=================================================================")
    print("   RUNNING ATTENUATION THEORY (SIM 03)                           ")
    print("=================================================================")
    compute_matsuda_ishii()
    compute_riccati_walk()
    compute_harmonic_envelope()
    compute_homogenization_bounds()
    compute_attenuation_methods()
    print("\n=================================================================")
    print("   SIM 03 COMPLETE!                                              ")
    print("=================================================================")
