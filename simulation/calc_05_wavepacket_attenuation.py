"""
calc_05_wavepacket_attenuation.py
=================================
Calculates the effective attenuation length of a Gaussian wavepacket (xi_wp)
across the propagating frequency band. It uses Exact Diagonalization to project
the initial wavepacket profile onto the exact eigenmodes, and extracts xi_wp
from the time-averaged Participation Ratio of the resulting energy distribution.

Outputs:
    - data/wavepacket_attenuation.npz
"""
import numpy as np
import os
import time
from scipy.linalg import eigh_tridiagonal

import sim_config
data_dir = sim_config.DATA_DIR

def compute_wp_attenuation():
    """
    Computes the wavepacket attenuation length for mass disorder, stiffness disorder,
    and combined disorder over a sweep of 15 carrier frequencies.
    Uses exact diagonalization on N=6000 atoms.
    """
    print("=================================================================")
    print("   RUNNING WAVEPACKET ATTENUATION (CALC 05)                      ")
    print("=================================================================")
    t0 = time.time()
    
    # 15 frequencies in the propagating band [0.1, 1.8]
    omega_vals = np.linspace(0.1, 1.8, 15)
    eps_val = 0.25
    N = 6000
    sigma_wp = 10.0  
    center = N // 2
    
    np.random.seed(42)
    
    eps_j_mass = eps_val * (np.random.rand(N) - 0.5) * 2.0
    masses = 1.0 * (1.0 + eps_j_mass)
    
    eps_j_spring = eps_val * (np.random.rand(N + 1) - 0.5) * 2.0
    springs = 1.0 * (1.0 + eps_j_spring)
    
    results = {
        "mass": {"xi_wp": np.zeros_like(omega_vals)},
        "spring": {"xi_wp": np.zeros_like(omega_vals)},
        "combined": {"xi_wp": np.zeros_like(omega_vals)}
    }
    
    for sys_name in ["mass", "spring", "combined"]:
        print(f"\n  [WP Attenuation] Processing {sys_name} disorder...")
        m_sys = masses if sys_name in ["mass", "combined"] else np.ones(N)
        k_sys = springs if sys_name in ["spring", "combined"] else np.ones(N+1)
        
        d = (k_sys[:-1] + k_sys[1:]) / m_sys
        e = -k_sys[1:-1] / np.sqrt(m_sys[:-1] * m_sys[1:])
        
        print(f"    -> Diagonalizing {sys_name} dynamical matrix (N={N})...")
        eigvals, eigvecs = eigh_tridiagonal(d, e)
        Psi = eigvecs / np.sqrt(m_sys)[:, None]
        
        nodes = np.arange(N)
        
        for i, w in enumerate(omega_vals):
            a = 1.0
            qa0 = 2 * np.arcsin(np.clip(w / 2.0, 0, 0.999))
            q0 = qa0 / a
            
            u0 = np.exp(-(nodes - center)**2 / (2 * sigma_wp**2)) * np.cos(q0 * a * (nodes - center))
            u0 = u0 / np.linalg.norm(u0)
            
            c = eigvecs.T @ (u0 * np.sqrt(m_sys))
            
            c_sq = c**2
            u_sq_avg = 0.5 * (Psi**2 @ c_sq)
            
            pr = (np.sum(u_sq_avg)**2) / np.sum(u_sq_avg**2)
            xi_wp = pr / 3.0
            results[sys_name]["xi_wp"][i] = xi_wp
            
            if (i+1) % 5 == 0:
                print(f"    -> Freq {w:.2f} ({i+1}/{len(omega_vals)}) -> xi_wp = {xi_wp:.2f}")

    output_path = os.path.join(data_dir, "wavepacket_attenuation.npz")
    np.savez_compressed(output_path, 
             omega=omega_vals, 
             xi_wp_mass=results["mass"]["xi_wp"],
             xi_wp_spring=results["spring"]["xi_wp"],
             xi_wp_combined=results["combined"]["xi_wp"])
             
    print(f"\n  -> Done in {time.time()-t0:.2f}s. Saved: wavepacket_attenuation.npz")
    print("=================================================================")
    print("   CALC 05 COMPLETE!                                             ")
    print("=================================================================")

if __name__ == '__main__':
    compute_wp_attenuation()
