import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

# --- High-Level Config ---
N = 1000        # Chain length
M = 80          # Monte Carlo realizations per point
plt.rcParams.update({"font.family": "serif", "font.size": 10})

def fast_lyapunov_sweep(epsilon_range, omega_range, type='mass'):
    """Vectorized sweep over epsilon and omega for statistical mean and std."""
    K = 1.0
    m0 = 1.0
    mean_gamma = np.zeros((len(epsilon_range), len(omega_range)))
    std_gamma = np.zeros((len(epsilon_range), len(omega_range)))
    
    for i, eps in enumerate(epsilon_range):
        # Pre-generate ensemble for this epsilon
        np.random.seed(42)
        ensemble_eps = eps * (np.random.rand(M, N) - 0.5)
        
        for j, w in enumerate(omega_range):
            gammas = []
            for m_idx in range(M):
                R = 1.0 + 0j
                log_transfers = []
                e_realization = ensemble_eps[m_idx]
                
                # Optimized Riccati inner loop
                if type == 'mass':
                    m_arr = m0 * (1 + e_realization)
                    for n in range(N):
                        R = (2 - (m_arr[n] * w**2 / K)) - 1/R
                        log_transfers.append(np.log(np.abs(R)))
                else:
                    k_arr = K * (1 + e_realization)
                    for n in range(N):
                        kn_prev = k_arr[n-1] if n > 0 else K
                        kn = k_arr[n]
                        R = ((kn + kn_prev - m0*w**2)/kn) - (kn_prev/kn)/R
                        log_transfers.append(np.log(np.abs(R)))
                
                gammas.append(np.mean(log_transfers))
            
            mean_gamma[i, j] = np.mean(gammas)
            std_gamma[i, j] = np.std(gammas)
            
    return mean_gamma, std_gamma

def run_comprehensive_audit():
    eps_vals = np.linspace(0.01, 1.2, 25)
    omega_vals = np.linspace(0.1, 2.2, 30)
    
    print("Starting Comprehensive Statistical Audit (Large Scale MC)...")
    m_mean, m_std = fast_lyapunov_sweep(eps_vals, omega_vals, type='mass')
    s_mean, s_std = fast_lyapunov_sweep(eps_vals, omega_vals, type='stiff')
    
    # --- Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.25, wspace=0.15)
    
    # Panel A: Mass Disorder Heatmap
    im1 = axes[0, 0].pcolormesh(omega_vals, eps_vals, m_mean, shading='gouraud', cmap='magma', norm=colors.LogNorm(vmin=1e-3, vmax=2))
    axes[0, 0].set_title("A. Plan de Localisation (Désordre de Masse)", fontweight='bold')
    axes[0, 0].set_ylabel("Intensité du désordre $\epsilon$")
    fig.colorbar(im1, ax=axes[0, 0], label=r"$\langle \gamma \rangle$ (Localisation)")
    
    # Panel B: Stiffness Disorder Heatmap
    im2 = axes[0, 1].pcolormesh(omega_vals, eps_vals, s_mean, shading='gouraud', cmap='magma', norm=colors.LogNorm(vmin=1e-3, vmax=2))
    axes[0, 1].set_title("B. Plan de Localisation (Désordre de Raideur)", fontweight='bold')
    fig.colorbar(im2, ax=axes[0, 1], label=r"$\langle \gamma \rangle$")
    
    # Panel C: Statistical Universality (Relative Variance)
    # Shows where the seed-to-seed fluctuations are highest
    rel_std = s_std / (s_mean + 1e-9)
    im3 = axes[1, 0].pcolormesh(omega_vals, eps_vals, rel_std, shading='gouraud', cmap='viridis')
    axes[1, 0].set_title("C. Sensibilité Statistique ($\sigma_{\gamma} / \langle \gamma \rangle$)", fontweight='bold')
    axes[1, 0].set_xlabel("Pulsation $\omega$")
    axes[1, 0].set_ylabel("$\epsilon$")
    fig.colorbar(im3, ax=axes[1, 0], label="Variance Relative")
    
    # Panel D: Evolution of the Localization Length (Cross-section)
    w_idx = len(omega_vals) // 2
    axes[1, 1].errorbar(eps_vals, 1/(m_mean[:, w_idx] + 1e-9), yerr=m_std[:, w_idx]/(m_mean[:, w_idx]**2 + 1e-9), fmt='o-', label='Masse', color='#1f77b4')
    axes[1, 1].errorbar(eps_vals, 1/(s_mean[:, w_idx] + 1e-9), yerr=s_std[:, w_idx]/(s_mean[:, w_idx]**2 + 1e-9), fmt='s-', label='Raideur', color='#d62728')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title(f"D. Longueur de Localisation $\\xi$ ($\omega = {omega_vals[w_idx]:.1f}$)", fontweight='bold')
    axes[1, 1].set_xlabel("$\epsilon$")
    axes[1, 1].set_ylabel(r"Distance de Propagation $\xi$ (atomes)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    save_path = "comprehensive_disorder_audit.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Comprehensive audit complete. Figure saved to {save_path}")
    # plt.show()

if __name__ == "__main__":
    run_comprehensive_audit()
