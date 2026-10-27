"""
Analyse 02 — Convergence Statistique (Effet du Seed / Nombre de Réalisations).

Étudie la convergence de l'estimateur de la longueur de pénétration L_p
en fonction du nombre de réalisations stochastiques M.
Vérifie la loi de convergence en 1/√M.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from engine.chain import DisorderedChain
from engine.style import apply_style, COLORS

apply_style()

# =============================================================================
# Configuration
# =============================================================================
N = 1500
EPSILON = 0.5
DISORDER_TYPE = 'stiff'
OMEGA_RATIO = 0.25
MAX_SEEDS = 50                # Nombre maximum de réalisations à tester
M_VALUES = [2, 5, 10, 15, 20, 30, 40, 50]

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("ANALYSE 02 : Convergence Statistique sur la Longueur de Pénétration")
    print("=" * 70)
    
    # Run all MAX_SEEDS simulations once
    print(f"Calcul de {MAX_SEEDS} réalisations (N={N}, ε={EPSILON}, type={DISORDER_TYPE})...")
    all_lp_values = []
    
    for seed in range(MAX_SEEDS):
        chain = DisorderedChain(N=N, epsilon=EPSILON, disorder_type=DISORDER_TYPE, seed=seed)
        result = chain.simulate(excitation='wavepacket', omega_ratio=OMEGA_RATIO,
                                steps=40000, downsample=200)
        lp = chain.extract_penetration_length(result['envelope'], result['excitation_node'])
        all_lp_values.append(lp)
        if (seed + 1) % 10 == 0:
            print(f"  ... {seed + 1}/{MAX_SEEDS} terminées")
    
    all_lp_values = np.array(all_lp_values)
    
    # Compute convergence: for each M, compute mean and std of the MEAN estimator
    means_of_means = []
    stds_of_means = []
    
    n_bootstrap = 200  # Bootstrap resampling for error estimation
    for M in M_VALUES:
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(all_lp_values, size=M, replace=True)
            bootstrap_means.append(np.mean(sample))
        means_of_means.append(np.mean(bootstrap_means))
        stds_of_means.append(np.std(bootstrap_means))
    
    means_of_means = np.array(means_of_means)
    stds_of_means = np.array(stds_of_means)
    
    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Convergence of the mean estimator
    reference_mean = np.mean(all_lp_values)
    ax1.axhline(reference_mean, color=COLORS['theory'], ls='--', alpha=0.4,
                label=fr'$L_{{p, ref}}$ (M={MAX_SEEDS})')
    ax1.errorbar(M_VALUES, means_of_means, yerr=stds_of_means,
                 fmt='o-', color=COLORS['stiff'], capsize=4, markersize=6,
                 label=r'Moyenne bootstrap $\langle L_p \rangle_M$')
    ax1.fill_between(M_VALUES,
                     means_of_means - stds_of_means,
                     means_of_means + stds_of_means,
                     color=COLORS['stiff'], alpha=0.15)
    
    ax1.set_xlabel(r"Nombre de r\'{e}alisations $M$")
    ax1.set_ylabel(r"Longueur de p\'{e}n\'{e}tration $L_p$ (sites)")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # Panel 2: Error scaling (log-log)
    relative_errors = stds_of_means / means_of_means
    ax2.loglog(M_VALUES, relative_errors, 'o-', color=COLORS['stiff'],
               markersize=6, label=r'Erreur relative $\sigma / \langle L_p \rangle$')
    
    # Theoretical 1/sqrt(M)
    M_arr = np.array(M_VALUES, dtype=float)
    scale = relative_errors[0] * np.sqrt(M_VALUES[0])
    ax2.loglog(M_arr, scale / np.sqrt(M_arr), '--', color=COLORS['theory'],
               alpha=0.5, label=r'Loi en $1/\sqrt{M}$')
    
    ax2.set_xlabel(r"Nombre de r\'{e}alisations $M$")
    ax2.set_ylabel(r"Erreur relative")
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.2)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "analyse_02_seed_convergence.pdf")
    plt.savefig(save_path)
    print(f"--> Figure sauvegardée : {save_path}")


if __name__ == '__main__':
    main()
