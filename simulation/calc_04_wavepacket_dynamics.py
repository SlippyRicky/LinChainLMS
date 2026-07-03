"""
calc_04_wavepacket_dynamics.py
==============================
Simulates the time-domain transient dynamics of wavepackets and impact events
propagating through a disordered chain. Uses both exact diagonalization for small
chains and Velocity Verlet integration for large thermodynamic chains (N=6000).

Outputs:
    - data/wavepacket_arrest_N400.npz
    - data/wp_vs_impact_comparison.npz
"""
import numpy as np
import os
import time
from scipy.linalg import eigh_tridiagonal

import sim_config
data_dir = sim_config.DATA_DIR

def compute_wavepacket_exact_diag():
    """
    Simulates the time evolution of a Gaussian wavepacket on a small chain (N=400)
    using exact diagonalization. The initial spatial profile is projected onto the 
    eigenmodes, and the time history is analytically advanced using cos(omega * t).
    """
    print("\n[Wavepacket Dynamics] Computing WP Arrest via Exact Diag (N=400)...")
    t0 = time.time()
    N = 400
    T_max = 600
    dt = 1.0
    times = np.arange(0, T_max, dt)
    
    np.random.seed(42)
    epsilon = 1.0 
    m_n = 1.0 * (1 + epsilon * (np.random.rand(N) - 0.5))
    K = 1.0
    
    # Fast tridiagonal exact diag
    d = np.zeros(N)
    e = np.zeros(N-1)
    for i in range(N):
        kl = K if i > 0 else 0
        kr = K if i < N-1 else 0
        d[i] = (kl + kr) / m_n[i]
        if i < N-1: e[i] = -K / np.sqrt(m_n[i] * m_n[i+1])
            
    eigvals, Psi = eigh_tridiagonal(d, e)
    omega = np.sqrt(np.clip(eigvals, 1e-9, None))
    
    sigma = 5.0
    q = np.pi / 2  
    nodes = np.arange(N)
    u0 = np.exp(-(nodes - N//2)**2 / (2 * sigma**2)) * np.cos(q * (nodes - N//2))
    u0 = u0 / np.linalg.norm(u0)
    
    coeffs = Psi.T @ u0
    
    data = np.zeros((len(times), N))
    for i, t in enumerate(times):
        u_t = Psi @ (coeffs * np.cos(omega * t))
        energy = u_t**2
        data[i, :] = np.convolve(energy, np.ones(5)/5, mode='same')
        
    filename = "wavepacket_arrest_N400.npz"
    np.savez(os.path.join(data_dir, filename), data=data, times=times, N=N, T_max=T_max)
    print(f"  -> Done in {time.time()-t0:.2f}s. Saved: {filename}")

def compute_comparative_dynamics_verlet():
    """
    Simulates the time evolution of three transient scenarios on a large chain (N=6000)
    using Velocity Verlet integration:
    1. A single node impact (delta impulse)
    2. A low-frequency wavepacket (q = 0.15*pi)
    3. A high-frequency wavepacket (q = 0.70*pi)
    
    Simulates the time evolution of a transient wavepacket on a large chain (N=6000)
    using Velocity Verlet integration for 3 different disorder levels.
    """
    print("\n[Wavepacket Dynamics] Computing Comparative Dynamics via Velocity Verlet (N=6000)...")
    t0 = time.time()
    
    N = 6000
    T_max = 1000
    dt = 0.05
    steps = int(T_max / dt)  
    downsample = 40          
    n_frames = steps // downsample
    times = np.arange(n_frames) * downsample * dt
    
    K = 1.0
    k_bonds = np.ones(N + 1) * K
    
    l_abc = 100
    gamma_max = 0.20
    g_profile = np.zeros(N)
    for i in range(l_abc):
        g_profile[i] = gamma_max * ((l_abc - i) / l_abc)**2
        g_profile[N-1-i] = gamma_max * ((l_abc - i) / l_abc)**2
        
    center = N // 2
    
    epsilons = [0.1, 0.3, 1.4]
    results = {}
    
    for eps in epsilons:
        print(f"  -> Simulating disorder epsilon: {eps}...")
        np.random.seed(42)
        m0 = 1.0
        m_n = m0 * (1.0 + eps * (np.random.rand(N) - 0.5))
        
        u = np.zeros(N)
        nodes = np.arange(N)
        sigma = 10.0
        q = 0.50 * np.pi
        u0 = np.exp(-(nodes - center)**2 / (2 * sigma**2)) * np.cos(q * (nodes - center))
        u0 = u0 / np.linalg.norm(u0)
        u[:] = u0
        
        v = np.zeros(N)
        
        def get_acc(u_c, v_c):
            f = np.zeros(N)
            f[0] = -k_bonds[0]*u_c[0] - k_bonds[1]*(u_c[0]-u_c[1])
            f[1:N-1] = -k_bonds[1:N-1]*(u_c[1:N-1]-u_c[0:N-2]) - k_bonds[2:N]*(u_c[1:N-1]-u_c[2:N])
            f[N-1] = -k_bonds[N-1]*(u_c[N-1]-u_c[N-2]) - k_bonds[N]*u_c[N-1]
            f -= g_profile * v_c
            return f / m_n
            
        a = get_acc(u, v)
        data = np.zeros((n_frames, N))
        
        for step in range(steps):
            v_half = v + 0.5 * dt * a
            u += v_half * dt
            
            a = get_acc(u, v_half)
            v = v_half + 0.5 * dt * a
            
            if step % downsample == 0:
                idx = step // downsample
                if idx < n_frames:
                    energy = u**2
                    data[idx, :] = np.convolve(energy, np.ones(5)/5, mode='same')
                    
        results[f"eps_{eps}"] = data

    filename = "wp_vs_impact_comparison.npz"
    np.savez(os.path.join(data_dir, filename), 
             eps_0_1=results["eps_0.1"], eps_0_3=results["eps_0.3"], 
             eps_1_4=results["eps_1.4"], times=times, N=N, T_max=T_max)
    print(f"  -> Done in {time.time()-t0:.2f}s. Saved: {filename}")

if __name__ == '__main__':
    print("=================================================================")
    print("   RUNNING WAVEPACKET DYNAMICS (CALC 04)                         ")
    print("=================================================================")
    compute_wavepacket_exact_diag()
    compute_comparative_dynamics_verlet()
    print("\n=================================================================")
    print("   CALC 04 COMPLETE!                                             ")
    print("=================================================================")
