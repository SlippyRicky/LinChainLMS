import numpy as np
import matplotlib.pyplot as plt
import os
import time
from multiprocessing import Pool

# ==========================================
# --- Configuration & Mode Selection ---
# ==========================================
# Choose the active disorder type:
# - 'stiff'   : Disorder only on springs (stiffness K_n)
# - 'mass'    : Disorder only on masses (m_n)
# - 'combined': Independent disorder on both masses and springs
DISORDER_MODE = 'stiff' 

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "grid.alpha": 0.3
})

def compute_lyapunov_ensemble(N, omega, epsilon, type='stiff', realizations=100):
    """
    Compute ensemble-averaged Lyapunov exponent using a highly optimized vectorized NumPy method.
    
    Supports:
    - 'stiff'   : stiffness fluctuations only (mass is homogeneous)
    - 'mass'    : mass fluctuations only (stiffness is homogeneous)
    - 'combined': independent fluctuations on both masses and stiffnesses
    """
    K = 1.0
    m0 = 1.0
    
    # Generate stochastics for all realizations at once (N, realizations)
    np.random.seed(int(epsilon * 10000))
    
    # Pre-allocate fluctuations
    eps_m = np.zeros((N, realizations))
    eps_k = np.zeros((N, realizations))
    
    if type == 'mass':
        eps_m = epsilon * (np.random.rand(N, realizations) - 0.5)
    elif type == 'stiff':
        eps_k = epsilon * (np.random.rand(N, realizations) - 0.5)
    elif type == 'combined':
        # Independent random paths for mass and stiffness disorder
        eps_m = epsilon * (np.random.rand(N, realizations) - 0.5)
        eps_k = epsilon * (np.random.rand(N, realizations) - 0.5)
        
    # State variable R is a vector of size (realizations,)
    R = np.ones(realizations, dtype=complex)
    log_transfers = np.zeros((N, realizations))
    
    for i in range(N):
        m_n = m0 * (1.0 + eps_m[i, :])
        kn = K * (1.0 + eps_k[i, :])
        kn_prev = K * (1.0 + eps_k[i-1, :]) if i > 0 else K
        
        # Generalised Riccati-like recurrence relation for mass-spring chains:
        R = ((kn + kn_prev - m_n * omega**2) / kn) - (kn_prev / kn) / R
        log_transfers[i, :] = np.log(np.abs(R))
        
    # Spatial average of log-transfers over the chain
    gammas = np.mean(log_transfers, axis=0)
    return np.mean(gammas), np.std(gammas)

def process_single_point(args):
    """Worker function for multiprocessing."""
    N, omega, eps, M, mode = args
    g, s = compute_lyapunov_ensemble(N, omega, eps, type=mode, realizations=M)
    return eps, g, s

def run_sensitivity_analysis():
    N = 1000
    M = 100        # High-fidelity Monte Carlo realizations
    omega = 0.5 
    epsilons = np.linspace(0.05, 0.8, 100) # Dense 100 points sweep
    
    # Aesthetic dictionary mapping mode -> (label, color)
    mode_aesthetics = {
        'stiff': ('Désordre de Raideur (Ressorts)', '#d62728'),
        'mass': ('Désordre de Masse', '#1f77b4'),
        'combined': ('Désordre Combiné (Masse + Raideur)', '#9467bd')
    }
    mode_label, mode_color = mode_aesthetics[DISORDER_MODE]
    
    print(f"Starting Highly Optimized Combined Monte Carlo Sweep...")
    print(f"Active Mode: {DISORDER_MODE.upper()} ({mode_label})")
    print(f"Parameters : N={N} sites | M={M} realizations | {len(epsilons)} Sweep Points")
    print(f"Utilizing NumPy Vectorization + Multiprocessing Parallel Core Sweeps...")
    
    # Prepare arguments for multiprocessing
    tasks = [(N, omega, eps, M, DISORDER_MODE) for eps in epsilons]
    
    # Measure execution time
    start_time = time.time()
    with Pool() as pool:
        results = pool.map(process_single_point, tasks)
    elapsed = time.time() - start_time
    
    print(f"--> Parallel sweep complete in {elapsed:.2f} seconds!")
    
    # Unpack and sort by epsilon (Pool keeps order, but sorting guarantees it)
    sorted_results = sorted(results, key=lambda x: x[0])
    eps_vals, gamma_mean, gamma_std = zip(*sorted_results)
    
    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Panel 1: Scaling Law with Error Bars
    ax1.errorbar(eps_vals, gamma_mean, yerr=gamma_std, fmt='o', color=mode_color, 
                 label=f'{mode_label} (Moyenne MC)', capsize=2, elinewidth=0.5, markersize=4, alpha=0.8)
    
    # Quadratic fit
    z = np.polyfit(eps_vals, gamma_mean, 2)
    p = np.poly1d(z)
    ax1.plot(eps_vals, p(eps_vals), '--', color='black', alpha=0.6, label=r'Loi de Matsuda-Ishii $\propto \epsilon^2$')
    
    ax1.set_title(f"Sensibilité : {mode_label}", fontweight='bold')
    ax1.set_xlabel(r"Amplitude du désordre $\epsilon$")
    ax1.set_ylabel(r"Exposant de Lyapunov $\gamma$")
    ax1.legend()
    ax1.grid(True)
    
    # Panel 2: Physical Comparison of Decay Envelopes
    N_spatial = 500
    x = np.arange(N_spatial)
    
    # High vs Low disorder envelopes
    for e, ls in zip([0.1, 0.4], ['-', '--']):
        g, _ = compute_lyapunov_ensemble(N, omega, e, type=DISORDER_MODE, realizations=M)
        ax2.plot(x, np.exp(-g * x), ls, label=fr"$\epsilon = {e}$ ({mode_label})", color=mode_color)

    ax2.set_title("Comparaison des Profils d'Atténuation", fontweight='bold')
    ax2.set_xlabel("Distance $n$")
    ax2.set_ylabel(r"Enveloppe $|u_n/u_0|$")
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    save_path = "disorder_sensitivity_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"--> Analysis complete. High-resolution figure saved to {save_path}")

if __name__ == "__main__":
    run_sensitivity_analysis()
