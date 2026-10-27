"""
Analyse 04 — Effet de la Taille de la Chaîne sur la Longueur de Pénétration.

Étudie la dépendance de la longueur de pénétration L_p en fonction
de la taille du système N.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from engine.chain import DisorderedChain
from engine.style import apply_style, COLORS

apply_style()

# =============================================================================
# Configuration
# =============================================================================
CHAIN_LENGTHS = [200, 400, 600, 800, 1000, 1500, 2000, 3000]
EPSILON = 0.5
DISORDER_TYPE = 'stiff'
OMEGA_RATIO = 0.25
SEEDS = range(8)

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_point(args):
    """Worker: compute penetration length for one (N, seed)."""
    chain_N, seed = args
    # Scale simulation time with chain length so the wave has time to propagate
    steps = max(20000, int(chain_N * 30))
    l_abc = min(100, chain_N // 6)
    
    chain = DisorderedChain(N=chain_N, epsilon=EPSILON, disorder_type=DISORDER_TYPE,
                            seed=seed, m0=1.0, k0=1.0)
    result = chain.simulate(excitation='wavepacket', omega_ratio=OMEGA_RATIO,
                            steps=steps, downsample=200, l_abc=l_abc)
    lp = chain.extract_penetration_length(result['envelope'], result['excitation_node'])
    return chain_N, seed, lp


def main():
    print("=" * 70)
    print("ANALYSE 04 : Effet de la Taille de la Chaîne sur la Longueur de Pénétration")
    print("=" * 70)
    
    tasks = [(chain_N, seed)
             for chain_N in CHAIN_LENGTHS
             for seed in SEEDS]
    
    print(f"Paramètres : ε={EPSILON} | {len(CHAIN_LENGTHS)} tailles N | {len(SEEDS)} seeds")
    print(f"Total : {len(tasks)} simulations")
    
    with Pool() as pool:
        results = pool.map(compute_point, tasks)
    
    # Organize: {N: [penetration_lengths]}
    data = {cn: [] for cn in CHAIN_LENGTHS}
    for chain_N, seed, lp in results:
        data[chain_N].append(lp)
    
    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    means = [np.mean(data[cn]) for cn in CHAIN_LENGTHS]
    stds  = [np.std(data[cn])  for cn in CHAIN_LENGTHS]
    
    # Panel 1: L_p vs N
    ax1.errorbar(CHAIN_LENGTHS, means, yerr=stds, fmt='o-',
                 color=COLORS['stiff'], capsize=4, markersize=6,
                 label=fr'$\epsilon = {EPSILON}$, {DISORDER_TYPE}')
    
    # Asymptotic reference
    if len(means) > 2:
        asymptotic = np.mean(means[-3:])
        ax1.axhline(asymptotic, color=COLORS['theory'], ls='--', alpha=0.4,
                    label=fr'$L_{{p, \infty}} \approx {asymptotic:.0f}$ sites')
    
    ax1.set_xlabel(r"Taille de la cha\^{i}ne $N$ (sites)")
    ax1.set_ylabel(r"Longueur de p\'{e}n\'{e}tration $L_p$ (sites)")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # Panel 2: Ratio L_p/N
    ratios = [np.mean(data[cn]) / cn for cn in CHAIN_LENGTHS]
    ratio_stds = [np.std(data[cn]) / cn for cn in CHAIN_LENGTHS]
    
    ax2.errorbar(CHAIN_LENGTHS, ratios, yerr=ratio_stds, fmt='s-',
                 color=COLORS['accent2'], capsize=4, markersize=6,
                 label=r'$L_p / N$')
    ax2.axhline(0, color='gray', ls='-', alpha=0.3)
    
    ax2.set_xlabel(r"Taille de la cha\^{i}ne $N$ (sites)")
    ax2.set_ylabel(r"Ratio $L_p / N$")
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "analyse_04_chain_length.pdf")
    plt.savefig(save_path)
    print(f"--> Figure sauvegardée : {save_path}")


if __name__ == '__main__':
    main()
