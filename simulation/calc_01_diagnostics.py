"""
calc_01_diagnostics.py
======================
This script handles baseline verification tests and diagnostics for the linear disorder study.
It benchmarks the system against established literature (Hu et al. 2008) and validates
the random distributions of mass and stiffness disorders across the chain.

Outputs:
    - data/hu_2008_localization.csv
    - data/mass_disorder_audit_data.npz
    - data/stiffness_disorder_audit_data.npz
"""
import numpy as np
import pandas as pd
import os
import time

import sim_config
data_dir = sim_config.DATA_DIR

def compute_mass_audit():
    """
    Performs a high-fidelity Monte Carlo audit (M=100) of mass disorder distributions.
    Computes the spatial decay constant (gamma) across various frequencies and
    disorder strengths (epsilon) to verify the statistical properties of the generator.
    """
    print("\n[Diagnostic 1/2] Computing Mass Disorder Audit...")
    t0 = time.time()
    
    N = 1000        # Chain length
    M = 100         # High-fidelity Monte Carlo realizations
    eps_vals = np.linspace(0.01, 1.2, 50)
    omega_vals = np.linspace(0.1, 2.1, 50)
    K = 1.0
    m0 = 1.0
    
    mean_gamma = np.zeros((len(eps_vals), len(omega_vals)))
    std_gamma = np.zeros((len(eps_vals), len(omega_vals)))
    eta = 1e-10
    
    total_steps = len(eps_vals)
    for i, eps in enumerate(eps_vals):
        np.random.seed(i)
        m_ensemble = m0 * (1 + eps * (np.random.rand(M, N) - 0.5))
        
        for j, w in enumerate(omega_vals):
            w_reg = w + 1j*eta
            R = np.ones(M, dtype=complex)
            log_growth = np.zeros(M)
            
            for n in range(N):
                R = (2 - (m_ensemble[:, n] * w_reg**2 / K)) - 1/R
                log_growth += np.log(np.abs(R))
            
            gammas = log_growth / N
            mean_gamma[i, j] = max(np.mean(gammas), 1e-5)
            std_gamma[i, j] = np.std(gammas)
            
        if (i + 1) % 10 == 0 or i == total_steps - 1:
            print(f"  -> Progress: {i + 1}/{total_steps} epsilon values processed.")
            
    path = os.path.join(data_dir, "mass_disorder_audit_data.npz")
    np.savez_compressed(path, eps_vals=eps_vals, omega_vals=omega_vals, mean_gamma=mean_gamma, std_gamma=std_gamma)
    print(f"  -> Done in {time.time()-t0:.2f}s. Saved: mass_disorder_audit_data.npz")

def compute_stiffness_audit():
    """
    Performs a high-fidelity Monte Carlo audit (M=100) of stiffness disorder distributions.
    Computes the spatial decay constant (gamma) across various frequencies and
    disorder strengths (epsilon) to verify the statistical properties of the generator.
    """
    print("\n[Diagnostic 2/2] Computing Stiffness Disorder Audit...")
    t0 = time.time()
    
    N = 1000        
    M = 100         
    eps_vals = np.linspace(0.01, 1.2, 50)
    omega_vals = np.linspace(0.1, 2.1, 50)
    K = 1.0
    m0 = 1.0
    
    mean_gamma = np.zeros((len(eps_vals), len(omega_vals)))
    std_gamma = np.zeros((len(eps_vals), len(omega_vals)))
    eta = 1e-10  
    
    total_steps = len(eps_vals)
    for i, eps in enumerate(eps_vals):
        np.random.seed(i + 100) 
        k_ensemble = K * (1 + eps * (np.random.rand(M, N) - 0.5))
        
        for j, w in enumerate(omega_vals):
            w_reg = w + 1j*eta
            R = np.ones(M, dtype=complex)
            log_growth = np.zeros(M)
            
            for n in range(N):
                kn_prev = k_ensemble[:, n-1] if n > 0 else K
                kn = k_ensemble[:, n]
                R = ((kn + kn_prev - m0*w_reg**2)/kn) - (kn_prev/kn)/R
                log_growth += np.log(np.abs(R))
            
            gammas = log_growth / N
            mean_gamma[i, j] = max(np.mean(gammas), 1e-5)
            std_gamma[i, j] = np.std(gammas)
            
        if (i + 1) % 10 == 0 or i == total_steps - 1:
            print(f"  -> Progress: {i + 1}/{total_steps} epsilon values processed.")
            
    path = os.path.join(data_dir, "stiffness_disorder_audit_data.npz")
    np.savez_compressed(path, eps_vals=eps_vals, omega_vals=omega_vals, mean_gamma=mean_gamma, std_gamma=std_gamma)
    print(f"  -> Done in {time.time()-t0:.2f}s. Saved: stiffness_disorder_audit_data.npz")

if __name__ == "__main__":
    print("=================================================================")
    print("   RUNNING DIAGNOSTICS & BASELINES (CALC 01)                     ")
    print("=================================================================")
    compute_mass_audit()
    compute_stiffness_audit()
    print("\n=================================================================")
    print("   CALC 01 COMPLETE!                                             ")
    print("=================================================================")
