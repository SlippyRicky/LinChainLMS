import numpy as np
import time

def compute_lyapunov_original(N, omega, epsilon, type='mass', realizations=50):
    K = 1.0
    m0 = 1.0
    gammas = []
    
    for seed in range(realizations):
        np.random.seed(seed)
        R = 1.0 + 0j
        log_transfers = []
        eps = epsilon * (np.random.rand(N) - 0.5)
        
        for i in range(N):
            if type == 'mass':
                m_n = m0 * (1 + eps[i])
                R = (2 - (m_n * omega**2 / K)) - 1/R
            else:
                kn = K * (1 + eps[i])
                kn_prev = K * (1 + eps[i-1]) if i > 0 else K
                R = ( (kn + kn_prev - m0*omega**2)/kn ) - (kn_prev/kn)/R
            log_transfers.append(np.log(np.abs(R)))
        gammas.append(np.mean(log_transfers))
        
    return np.mean(gammas), np.std(gammas)

def compute_lyapunov_vectorized(N, omega, epsilon, type='mass', realizations=50):
    K = 1.0
    m0 = 1.0
    
    # Pre-generate all stochastics for all realizations at once
    # Shape: (N, realizations)
    np.random.seed(42) # single seed for reproducibility of ensemble
    eps = epsilon * (np.random.rand(N, realizations) - 0.5)
    
    # State variable R is now a vector of size (realizations,)
    R = np.ones(realizations, dtype=complex)
    
    # Store log transfers
    log_transfers = np.zeros((N, realizations))
    
    for i in range(N):
        if type == 'mass':
            m_n = m0 * (1 + eps[i, :])
            R = (2 - (m_n * omega**2 / K)) - 1.0 / R
        else:
            kn = K * (1 + eps[i, :])
            kn_prev = K * (1 + eps[i-1, :]) if i > 0 else K
            R = ((kn + kn_prev - m0 * omega**2) / kn) - (kn_prev / kn) / R
        log_transfers[i, :] = np.log(np.abs(R))
        
    gammas = np.mean(log_transfers, axis=0)
    return np.mean(gammas), np.std(gammas)

if __name__ == "__main__":
    N = 1000
    M = 100
    omega = 0.5
    eps = 0.3
    
    print(f"Benchmarking Lyapunov Exponent Calculation (N={N}, M={M} realizations)...")
    
    # Original
    t0 = time.time()
    mean_orig, std_orig = compute_lyapunov_original(N, omega, eps, 'mass', M)
    t_orig = time.time() - t0
    print(f"Original Code:   {t_orig:.4f} seconds | Mean: {mean_orig:.6f}, Std: {std_orig:.6f}")
    
    # Vectorized
    t0 = time.time()
    mean_vec, std_vec = compute_lyapunov_vectorized(N, omega, eps, 'mass', M)
    t_vec = time.time() - t0
    print(f"Vectorized Code: {t_vec:.4f} seconds | Mean: {mean_vec:.6f}, Std: {std_vec:.6f}")
    
    speedup = t_orig / t_vec
    print(f"--> Speedup: {speedup:.1f}x faster!")
