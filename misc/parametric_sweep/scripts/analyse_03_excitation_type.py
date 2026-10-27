"""
Analyse 03 — Comparaison Paquet d'Onde vs Impact sur la Longueur de Pénétration.

Compare la longueur de pénétration L_p extraite pour deux types d'excitation :
    - Paquet d'onde (burst gaussien monochromatique à ω/ω_max = 0.25)
    - Impact (impulsion de Dirac large bande)
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
N = 1500
SEEDS = range(10)
EPSILONS = np.linspace(0.1, 0.9, 12)
DISORDER_TYPE = 'stiff'
OMEGA_RATIO = 0.25
EXCITATION_TYPES = ['wavepacket', 'impact']

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aesthetic mapping
EXC_STYLE = {
    'wavepacket': {'color': COLORS['stiff'],   'marker': 'o',
                   'label': r'Paquet d\'onde ($\omega/\omega_{\max}=0.25$)'},
    'impact':     {'color': COLORS['accent1'], 'marker': 's',
                   'label': r'Impact (large bande)'},
}


def compute_point(args):
    """Worker: compute penetration length for one (epsilon, excitation, seed)."""
    eps, exc_type, seed = args
    chain = DisorderedChain(N=N, epsilon=eps, disorder_type=DISORDER_TYPE, seed=seed)
    result = chain.simulate(excitation=exc_type, omega_ratio=OMEGA_RATIO,
                            steps=40000, downsample=200)
    lp = chain.extract_penetration_length(result['envelope'], result['excitation_node'])
    return eps, exc_type, seed, lp


def main():
    print("=" * 70)
    print("ANALYSE 03 : Longueur de Pénétration (Paquet d'Onde vs Impact)")
    print("=" * 70)
    
    tasks = [(eps, exc, seed)
             for eps in EPSILONS
             for exc in EXCITATION_TYPES
             for seed in SEEDS]
    
    print(f"Paramètres : N={N} | {len(EPSILONS)} valeurs ε | "
          f"{len(SEEDS)} seeds | 2 types d'excitation")
    print(f"Total : {len(tasks)} simulations")
    
    with Pool() as pool:
        results = pool.map(compute_point, tasks)
    
    # Organize: {exc_type: {eps: [penetration_lengths]}}
    data = {exc: {eps: [] for eps in EPSILONS} for exc in EXCITATION_TYPES}
    for eps, exc_type, seed, lp in results:
        data[exc_type][eps].append(lp)
    
    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for exc_type in EXCITATION_TYPES:
        style = EXC_STYLE[exc_type]
        means = [np.mean(data[exc_type][eps]) for eps in EPSILONS]
        stds  = [np.std(data[exc_type][eps])  for eps in EPSILONS]
        
        ax1.errorbar(EPSILONS, means, yerr=stds, fmt=f'{style["marker"]}-',
                     color=style['color'], label=style['label'],
                     capsize=3, markersize=5)
    
    ax1.set_xlabel(r"Amplitude du d\'{e}sordre $\epsilon$")
    ax1.set_ylabel(r"Longueur de p\'{e}n\'{e}tration $L_p$ (sites)")
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # Panel 2: Envelope comparison for a fixed epsilon showing 1/e level
    eps_demo = 0.5
    seed_demo = 42
    for exc_type in EXCITATION_TYPES:
        style = EXC_STYLE[exc_type]
        chain = DisorderedChain(N=N, epsilon=eps_demo, disorder_type=DISORDER_TYPE,
                                seed=seed_demo)
        result = chain.simulate(excitation=exc_type, omega_ratio=OMEGA_RATIO,
                                steps=40000, downsample=200)
        env = result['envelope']
        node = result['excitation_node']
        
        # Normalize envelope to max = 1
        env_norm = env / np.max(env) if np.max(env) > 0 else env
        
        # Plot right side
        x = np.arange(len(env_norm) - node)
        ax2.semilogy(x, env_norm[node:], '-', color=style['color'],
                     label=style['label'], alpha=0.8)
    
    # Draw horizontal line at 1/e level
    ax2.axhline(1.0 / np.exp(1.0), color=COLORS['theory'], ls='--', alpha=0.6,
                label=r'Niveau $1/e \approx 0.368$')
    
    ax2.set_xlabel(r"Distance depuis l'excitation $|n - n_0|$ (sites)")
    ax2.set_ylabel(r"Enveloppe normalis\'{e}e $|u_n|/|u_{n_0}|$")
    ax2.set_xlim(0, N // 2)
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "analyse_03_excitation_type.pdf")
    plt.savefig(save_path)
    print(f"--> Figure sauvegardée : {save_path}")


if __name__ == '__main__':
    main()
