"""
Analyse 01 — Effet du Type de Désordre sur la Longueur de Pénétration.

Compare la longueur de pénétration L_p pour :
    - Désordre de raideur uniquement ('stiff')
    - Désordre de masse uniquement ('mass')
    - Désordre combiné ('combined')

sur une gamme d'amplitudes de désordre ε, à chaîne fixe (N=1500, ω/ω_max=0.25).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from engine.chain import DisorderedChain
from engine.style import apply_style, COLORS, DISORDER_LABELS

apply_style()

# =============================================================================
# Configuration
# =============================================================================
N = 1500
SEEDS = range(10)           # 10 réalisations pour la moyenne d'ensemble
EPSILONS = np.linspace(0.1, 0.9, 15)
OMEGA_RATIO = 0.25
DISORDER_TYPES = ['stiff', 'mass', 'combined']

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_point(args):
    """Worker: compute penetration length for one (epsilon, disorder_type, seed)."""
    eps, dtype, seed = args
    chain = DisorderedChain(N=N, epsilon=eps, disorder_type=dtype, seed=seed)
    result = chain.simulate(excitation='wavepacket', omega_ratio=OMEGA_RATIO,
                            steps=40000, downsample=200)
    lp = chain.extract_penetration_length(result['envelope'], result['excitation_node'])
    return eps, dtype, seed, lp


def main():
    print("=" * 70)
    print("ANALYSE 01 : Effet du Type de Désordre sur la Longueur de Pénétration")
    print("=" * 70)
    
    # Build task list
    tasks = [(eps, dtype, seed)
             for eps in EPSILONS
             for dtype in DISORDER_TYPES
             for seed in SEEDS]
    
    print(f"Paramètres : N={N} | {len(EPSILONS)} valeurs ε | "
          f"{len(SEEDS)} seeds | {len(DISORDER_TYPES)} types")
    print(f"Total : {len(tasks)} simulations")
    
    with Pool() as pool:
        results = pool.map(compute_point, tasks)
    
    # Organize results: {dtype: {eps: [penetration_lengths]}}
    data = {dt: {eps: [] for eps in EPSILONS} for dt in DISORDER_TYPES}
    for eps, dtype, seed, lp in results:
        data[dtype][eps].append(lp)
    
    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for dtype in DISORDER_TYPES:
        means = [np.mean(data[dtype][eps]) for eps in EPSILONS]
        stds  = [np.std(data[dtype][eps])  for eps in EPSILONS]
        
        ax1.errorbar(EPSILONS, means, yerr=stds, fmt='o-',
                     color=COLORS[dtype], label=DISORDER_LABELS[dtype],
                     capsize=3, markersize=5)
        
        # Inverse: attenuation coefficient = 1/L_p
        attenuations = [1.0 / m if m > 0 else 0 for m in means]
        ax2.plot(EPSILONS, attenuations, 's-', color=COLORS[dtype],
                 label=DISORDER_LABELS[dtype], markersize=5)
    
    # Quadratic reference
    ax2.plot(EPSILONS, 0.08 * EPSILONS**2, '--', color=COLORS['theory'],
             alpha=0.5, label=r'R\'{e}f. $\propto \epsilon^2$')
    
    ax1.set_xlabel(r"Amplitude du d\'{e}sordre $\epsilon$")
    ax1.set_ylabel(r"Longueur de p\'{e}n\'{e}tration $L_p$ (sites)")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    ax2.set_xlabel(r"Amplitude du d\'{e}sordre $\epsilon$")
    ax2.set_ylabel(r"Coefficient d'att\'{e}nuation $1/L_p$")
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "analyse_01_disorder_type.pdf")
    plt.savefig(save_path)
    print(f"--> Figure sauvegardée : {save_path}")


if __name__ == '__main__':
    main()
