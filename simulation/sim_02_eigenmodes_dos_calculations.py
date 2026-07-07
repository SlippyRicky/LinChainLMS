"""
sim_02_eigenmodes_dos_calculations.py
=====================================
Computes the fundamental modal properties of the disordered linear chain.
Uses fast tridiagonal exact diagonalization to compute eigenfrequencies (omega)
and eigenmodes (Psi) for both monatomic and diatomic chains.
Calculates Density of States (DOS) asymmetries and spatial heatmaps.

Outputs:
    - data/disordered_modes_*.npz
    - data/diatomic_disordered_modes_*.npz
    - data/dos_asymmetry_N*.npz
    - data/diatomic_modes.csv
"""
import numpy as np
import os
import time
from scipy.linalg import eigh_tridiagonal

import sim_config
data_dir = sim_config.DATA_DIR

def compute_modes(N, eps, is_diatomic=False):
    """
    Computes and saves the exact eigenfrequencies and eigenvectors for a disordered
    chain of size N and disorder strength eps.
    Optimized with scipy.linalg.eigh_tridiagonal for O(N^2) performance.
    """
    t0 = time.time()
    print(f"\n[Modal Analysis] Computing modes for N={N}, eps={eps}{' (diatomic)' if is_diatomic else ''}...")
    K = 1.0
    np.random.seed(42)
    if is_diatomic:
        m0 = np.array([1.0, 3.0] * (N // 2))
    else:
        m0 = 1.0
    masses = m0 * (1.0 + eps * (np.random.rand(N) - 0.5))
    
    K_arr = np.ones(N-1) * K
    
    # Fast Tridiagonal setup
    d = np.zeros(N)
    e = np.zeros(N-1)
    for i in range(N):
        kl = K_arr[i-1] if i > 0 else 0
        kr = K_arr[i] if i < N-1 else 0
        d[i] = (kl + kr) / masses[i]
        if i < N-1:
            e[i] = -K_arr[i] / np.sqrt(masses[i] * masses[i+1])
            
    eigvals, eigvecs = eigh_tridiagonal(d, e)
    omega = np.sqrt(np.clip(eigvals, 0, None))
    
    if is_diatomic:
        d_perf = np.zeros(N)
        for i in range(N):
            m_perf = m0[i]
            kl = K_arr[i-1] if i > 0 else 0
            kr = K_arr[i] if i < N-1 else 0
            d_perf[i] = (kl + kr) / m_perf
            
        eigvals_perf, _ = eigh_tridiagonal(d_perf, e)
        w_perf = np.sqrt(np.clip(eigvals_perf, 0, None))
    
    prefix = "diatomic_" if is_diatomic else ""
    filename = f"{prefix}disordered_modes_N{N}_eps{eps}.npz"
    if is_diatomic:
        np.savez_compressed(os.path.join(data_dir, filename), omega=omega, eigvecs=eigvecs, masses=masses, w_perf=w_perf)
    else:
        np.savez_compressed(os.path.join(data_dir, filename), omega=omega, eigvecs=eigvecs, masses=masses)
    print(f"  -> Done in {time.time()-t0:.2f}s. Saved: {filename}")

def compute_dos_asymmetry():
    """
    Computes the Density of States (DOS) for both stiffness disorder and mass disorder.
    Highlights the fundamental asymmetry: mass disorder preserves the upper bound omega_max,
    while stiffness disorder causes the spectrum to spread beyond the pristine band.
    """
    t0 = time.time()
    print("\n[Modal Analysis] Computing DOS Asymmetry...")
    
    params = sim_config.get_params("sim_02")
    N = params['N_asym_dos']
    
    np.random.seed(42)
    m_ref, K_ref = 1.0, 1.0
    epsilon = 0.5
    
    # Stiffness Disorder
    K_dis = np.random.uniform(K_ref*(1-0.5*epsilon), K_ref*(1+0.5*epsilon), N-1)
    M_fixed = np.ones(N) * m_ref
    d_K = np.zeros(N)
    e_K = np.zeros(N-1)
    for i in range(N):
        kl = K_dis[i-1] if i > 0 else 0
        kr = K_dis[i] if i < N-1 else 0
        d_K[i] = (kl + kr) / M_fixed[i]
        if i < N-1: e_K[i] = -K_dis[i] / np.sqrt(M_fixed[i] * M_fixed[i+1])
        
    eigvals_K, _ = eigh_tridiagonal(d_K, e_K)
    w_K = np.sqrt(np.clip(eigvals_K, 1e-9, None))
    
    # Mass Disorder
    M_dis = np.random.uniform(m_ref*(1-0.5*epsilon), m_ref*(1+0.5*epsilon), N)
    K_fixed = np.ones(N-1) * K_ref
    d_m = np.zeros(N)
    e_m = np.zeros(N-1)
    for i in range(N):
        kl = K_fixed[i-1] if i > 0 else 0
        kr = K_fixed[i] if i < N-1 else 0
        d_m[i] = (kl + kr) / M_dis[i]
        if i < N-1: e_m[i] = -K_fixed[i] / np.sqrt(M_dis[i] * M_dis[i+1])
        
    eigvals_m, _ = eigh_tridiagonal(d_m, e_m)
    w_m = np.sqrt(np.clip(eigvals_m, 1e-9, None))
    
    filename = "dos_asymmetry_N4000.npz"
    np.savez_compressed(os.path.join(data_dir, filename), w_K=w_K, w_m=w_m)
    print(f"  -> Done in {time.time()-t0:.2f}s. Saved: {filename}")

def compute_diatomic_modes_csv():
    """
    Generates idealized conceptual acoustic and optical mode shapes for a diatomic chain.
    Saved to CSV for schematic plotting in the report.
    """
    t0 = time.time()
    print("\n[Modal Analysis] Computing Diatomic Modes CSV...")
    N = 20
    m1, m2 = 1.0, 4.0
    sites = np.arange(N)
    
    u_acous = np.cos(0.1 * sites)
    
    u_optic = np.zeros(N)
    u_optic[0::2] = 1.0 # m1
    u_optic[1::2] = -m1/m2 # m2 
    
    data = np.column_stack((sites, u_acous, u_optic))
    path = os.path.join(data_dir, "diatomic_modes.csv")
    np.savetxt(path, data, delimiter=',')
    print(f"  -> Done in {time.time()-t0:.2f}s. Saved: diatomic_modes.csv")

if __name__ == '__main__':
    print("=================================================================")
    print("   RUNNING MODAL PROPERTIES (SIM 02)                             ")
    print("=================================================================")
    params = sim_config.get_params("sim_02")
    
    compute_modes(100, 0.9)
    compute_modes(500, 0.3)
    compute_modes(500, 0.4)
    compute_modes(500, 0.1)
    compute_modes(500, 0.7)
    compute_modes(500, 1.4)
    
    compute_modes(params['N_diatomic'], 0.4, is_diatomic=True)
    compute_dos_asymmetry()
    compute_diatomic_modes_csv()
    print("\n=================================================================")
    print("   SIM 02 COMPLETE!                                              ")
    print("=================================================================")
