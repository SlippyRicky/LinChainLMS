"""
calc_03_attenuation_theory.py
=============================
The central theory engine for the linear disorder study.
Aggregates all analytical and semi-analytical methods for computing the spatial
attenuation/localization length (xi) and the Lyapunov exponent (gamma).
Methods included:
    - Transfer Matrix Method (TMM)
    - Phase Formalism (Riccati approach)
    - Participation Ratio (Eigenmode projection)
    - Direct Harmonic Response (Green's function)
    - Effective Medium / Homogenization (Voigt/Reuss)

Outputs:
    - data/matsuda_ishii_scaling_multi.npz
    - data/mean_log_u_stiffness.csv
    - data/phase_diffusion_N200.npz
    - data/riccati_walk_N1000.npz
    - data/harmonic_envelope_N300.npz
    - data/homogenization_bounds_sweep.csv
    - data/attenuation_methods_comparison.npz
"""
import numpy as np
import pandas as pd
import os
import time
from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy.linalg import eigh_tridiagonal
from numba import njit

import sim_config
data_dir = sim_config.DATA_DIR

# =================================================================
# HELPER FUNCTIONS
# =================================================================
@njit
def compute_tmm_single_eps(omega_vals, eps, N, M):
    """
    Numba-accelerated core function to compute the Lyapunov exponent via
    the Transfer Matrix Method (TMM). Averages over M realizations of a chain of length N.
    """
    gamma_sim = np.zeros(len(omega_vals))
    for idx in range(len(omega_vals)):
        w = omega_vals[idx]
        w_sq = w**2
        omega_max = 2.0
        a = 1.0
        q_val = w / omega_max
        if q_val > 0.999: q_val = 0.999
        elif q_val < 0.001: q_val = 0.001
        qa = 2.0 * np.arcsin(q_val)
        q = qa / a
        
        realization_gammas = np.zeros(M)
        for m in range(M):
            springs = 1.0 + eps * (np.random.rand(N + 1) - 0.5)
            log_g = 0.0
            r = np.exp(1j * q * a)
            for n in range(N):
                r_next = 1.0 + springs[n]/springs[n+1] - w_sq/springs[n+1] - (springs[n]/springs[n+1]) / r
                if np.abs(r_next) < 1e-12: r_next = 1e-12
                log_g += np.log(np.abs(r_next))
                r = r_next
            realization_gammas[m] = np.abs(log_g / N)
        gamma_sim[idx] = np.mean(realization_gammas)
    return gamma_sim

def compute_linearized_riccati_gamma(N_sites, omega, eps_j):
    """
    Computes the Lyapunov exponent using the linearized Riccati (phase formalism) equation.
    Operates efficiently on a single array of disorder perturbations `eps_j`.
    """
    omega_max = 2.0
    a = 1.0
    qa = 2 * np.arcsin(np.clip(omega / omega_max, 0, 0.999))
    q = qa / a
    f_q = 2 * np.exp(-1j * q * a) * (1 - np.cos(q * a))
    
    A = -f_q * eps_j - 0.5 * (f_q**2) * (eps_j**2)
    B = np.exp(-2j * q * a) + f_q * eps_j
    
    B_norm = np.abs(B)
    B = np.where(B_norm > 0, B / B_norm, np.exp(-2j * q * a))
    
    B_cumprod = np.cumprod(B)
    terms_to_sum = A / B_cumprod
    
    inner_sum = np.zeros(N_sites, dtype=complex)
    inner_sum[1:] = np.cumsum(terms_to_sum[:-1])
    
    total_prod = np.ones(N_sites, dtype=complex)
    total_prod[1:] = B_cumprod[:-1]
    
    delta_eta = inner_sum * total_prod
    gamma = np.real(np.sum(delta_eta)) / N_sites
    return gamma


# =================================================================
# MAIN COMPUTATION BLOCKS
# =================================================================

def compute_matsuda_ishii():
    """
    Validates the Matsuda-Ishii analytical approximation (gamma ~ omega^2) against
    the exact Transfer Matrix Method (TMM) for low frequencies.
    """
    print("\n[Theory] Computing Matsuda-Ishii (TMM Scaling)...")
    t0 = time.time()
    omega_vals = np.logspace(-2, 0, 40)
    eps_values = [0.1, 0.3, 1.4]
    N = 50000
    M = 10
    
    results = {}
    np.random.seed(42)
    
    for eps in eps_values:
        sigma_sq = (eps**2) / 12.0
        gamma_theory = (sigma_sq / 8.0) * omega_vals**2 / (1.0 - omega_vals**2 / 4.0)
        gamma_sim = compute_tmm_single_eps(omega_vals, eps, N, M)
            
        results[f"gamma_theory_{eps}"] = gamma_theory
        results[f"gamma_sim_{eps}"] = gamma_sim
        print(f"    -> eps={eps} done")
        
    filename = "matsuda_ishii_scaling_multi.npz"
    np.savez(os.path.join(data_dir, filename), 
             omega=omega_vals, eps_values=np.array(eps_values), **results)
    print(f"  -> Done in {time.time()-t0:.2f}s. Saved: {filename}")

def compute_localization_decay():
    """
    Generates conceptual data demonstrating the exponential spatial decay of
    transmissions (mean log u) along the chain for various disorder strengths.
    """
    print("\n[Theory] Computing Localization Decay...")
    N = 1000
    eps_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    gamma_values = [0.001, 0.005, 0.015, 0.03, 0.05] 
    
    node_index = np.arange(N)
    df_log_u = pd.DataFrame({"NodeIndex": node_index})
    
    for i, eps in enumerate(eps_values):
        gamma = gamma_values[i]
        mean_log_u = -gamma * node_index + 0.1 * np.random.randn(N)
        df_log_u[f"MeanLogU_{eps}"] = mean_log_u
        
    df_eps = pd.DataFrame({"EpsilonValues": eps_values})
    
    path_log_u = os.path.join(data_dir, "mean_log_u_stiffness.csv")
    path_eps = os.path.join(data_dir, "eps_metadata_stiffness.csv")
    df_log_u.to_csv(path_log_u, index=False)
    df_eps.to_csv(path_eps, index=False)
    print("  -> Saved mean_log_u_stiffness.csv")

# def compute_phase_diffusion():
#     """
#     Simulates the spatial accumulation of phase perturbations (random walk of the phase)
#     which underpins the Riccati localization mechanism.
#     """
#     print("\n[Theory] Computing Phase Diffusion...")
#     np.random.seed(42) 
#     n_sites = 200
#     q0 = 0.15 * np.pi
#     n_realizations = 6
#     sigma_noise = 0.15 
    
#     phase_disordered_all = []
#     for i in range(n_realizations):
#         noise = np.random.normal(0, sigma_noise, n_sites)
#         phase_disordered = np.arange(n_sites) * q0 + np.cumsum(noise)
#         phase_disordered_all.append(phase_disordered)
        
#     filename = "phase_diffusion_N200.npz"
#     np.savez(os.path.join(data_dir, filename), 
#              phases=np.array(phase_disordered_all),
#              q0=q0, sigma_noise=sigma_noise)
#     print(f"  -> Saved {filename}")

def compute_riccati_walk():
    """
    Calculates a stochastic trajectory of the complex phase perturbation eta_n
    using the stabilized Riccati recurrence relation.
    """
    print("\n[Theory] Computing Riccati Walk...")
    N = 1000 
    eps_val = 0.4
    a = 1.0
    qa0 = 0.2 * np.pi 
    q0 = qa0 / a
    
    np.random.seed(42)
    eps = np.random.uniform(-eps_val, eps_val, N)
    
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
    np.savez(os.path.join(data_dir, filename), eta_walk=eta_walk, R=R, q0=q0 * a)
    print(f"  -> Saved {filename}")

def compute_harmonic_envelope():
    """
    Computes the steady-state spatial envelope of a harmonically driven chain
    using a long Velocity Verlet time integration, allowing transients to decay via damping.
    """
    print("\n[Theory] Computing Harmonic Envelope (Green's Function)...")
    t0 = time.time()
    N = 300
    epsilon_max = 0.3
    M0, K0 = 1.0, 1.0
    omega = np.sqrt(2)
    g = 0.01  
    
    np.random.seed(42)
    epsilon = np.random.uniform(-epsilon_max, epsilon_max, N)
    m = M0 * (1 + epsilon)
    K = np.ones(N + 1) * K0
    
    u = np.zeros(N)
    v = np.zeros(N)
    a = np.zeros(N)
    envelope = np.zeros(N)
    
    dt = 0.05
    STEPS = 60000 
    
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
        
        if step > STEPS - 15000:
            np.maximum(envelope, np.abs(u), out=envelope)

    nodes = np.arange(N)
    fit_mask = (nodes >= 10) & (nodes <= 150) & (envelope > 1e-6)
    fit_nodes = nodes[fit_mask]
    fit_u = envelope[fit_mask]
    
    slope, intercept, _, _, _ = linregress(fit_nodes, np.log(fit_u))
    gamma = -slope
    theory_env = np.exp(intercept - gamma * nodes)
    
    node_peaks = nodes
    u_peaks = envelope
    
    filename = "harmonic_envelope_N300.npz"
    np.savez(os.path.join(data_dir, filename), 
             nodes=nodes, u_max=envelope, 
             node_peaks=node_peaks, u_peaks=u_peaks, 
             theory_env=theory_env, gamma=gamma, N=N)
    print(f"  -> Done in {time.time()-t0:.2f}s. Saved: {filename}")

def compute_homogenization_bounds():
    """
    Calculates the Voigt (arithmetic mean) and Reuss (harmonic mean) theoretical bounds
    for the effective stiffness of a randomly disordered chain in the long-wavelength limit.
    """
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
    The master comparative script. Evaluates the spatial attenuation length (xi)
    across the entire frequency band using three parallel theoretical methods:
    1. Transfer Matrix Method (exact numerical)
    2. Linearized Riccati Formalism
    3. Participation Ratio of Exact Eigenmodes
    """
    print("\n[Theory] Computing Attenuation Methods Comparison...")
    t0 = time.time()
    omega_vals = np.linspace(0.05, 1.95, 200)
    eps_val = 0.25
    N = 6000 
    
    sigma_sq = (eps_val**2) / 3.0
    gamma_analytical = (sigma_sq * omega_vals**2) / (8.0 * (1.0 - (omega_vals**2) / 4.0))
    xi_analytical = 1.0 / gamma_analytical
    
    xi_tmm = np.zeros_like(omega_vals)
    xi_riccati = np.zeros_like(omega_vals)
    
    np.random.seed(42)
    eps_j = eps_val * (np.random.rand(N) - 0.5) * 2.0
    masses = 1.0 * (1.0 + eps_j)
    
    for i, w in enumerate(omega_vals):
        w_sq = w**2
        omega_max = 2.0
        a = 1.0
        qa = 2 * np.arcsin(np.clip(w / omega_max, 0, 0.999))
        q = qa / a
        log_g_tmm = 0.0
        r = np.exp(1j * q * a) 
        for n in range(N):
            r_next = (2.0 - masses[n] * w_sq) - 1.0 / r
            if np.abs(r_next) < 1e-12: r_next = 1e-12
            log_g_tmm += np.log(np.abs(r_next))
            r = r_next
        gamma_tmm = log_g_tmm / N
        xi_tmm[i] = 1.0 / np.abs(gamma_tmm) if np.abs(gamma_tmm) > 1e-12 else np.nan
        
        gamma_ric = compute_linearized_riccati_gamma(N, w, eps_j)
        xi_riccati[i] = 1.0 / np.abs(gamma_ric) if np.abs(gamma_ric) > 1e-12 else np.nan
        
        if (i+1) % 50 == 0:
            print(f"    -> Progress: {i+1}/{len(omega_vals)} frequencies")

    print("  -> Computing Participation Ratio (PR)...")
    N_PR = 6000 
    eps_j_pr = eps_val * (np.random.rand(N_PR) - 0.5) * 2.0
    masses_pr = 1.0 * (1.0 + eps_j_pr)
    
    d = 2.0 / masses_pr
    e = -1.0 / np.sqrt(masses_pr[:-1] * masses_pr[1:])
    
    eigvals, eigvecs = eigh_tridiagonal(d, e)
    omega_pr = np.sqrt(np.clip(eigvals, 0, None))
    
    u_sq = eigvecs**2
    pr = (np.sum(u_sq, axis=0)**2) / np.sum(u_sq**2, axis=0)
    xi_pr = pr / 3.0 

    filename = "attenuation_methods_comparison.npz"
    output_path = os.path.join(data_dir, filename)
    np.savez(output_path, 
             omega=omega_vals, xi_tmm=xi_tmm, xi_riccati=xi_riccati,
             xi_analytical=xi_analytical, epsilon=eps_val,
             omega_pr=omega_pr, xi_pr=xi_pr)
             
    print(f"  -> Done in {time.time()-t0:.2f}s. Saved: {filename}")

if __name__ == '__main__':
    print("=================================================================")
    print("   RUNNING ATTENUATION THEORY (CALC 03)                          ")
    print("=================================================================")
    compute_matsuda_ishii()
    compute_localization_decay()
    # compute_phase_diffusion()
    compute_riccati_walk()
    compute_harmonic_envelope()
    compute_homogenization_bounds()
    compute_attenuation_methods()
    print("\n=================================================================")
    print("   CALC 03 COMPLETE!                                             ")
    print("=================================================================")
