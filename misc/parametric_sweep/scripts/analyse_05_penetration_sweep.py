"""
Analyse 05 — Étude Paramétrique de la Longueur de Pénétration L_p.

Calcule la longueur de pénétration L_p en fonction de :
    - La fréquence d'excitation normalisée (omega / omega_max)
    - L'amplitude du désordre (epsilon)

Définition de L_p : La distance à laquelle le déplacement maximum (l'enveloppe)
atteint 1/e de sa valeur au point d'excitation (source).
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
# Configuration du Balayage 2D
# =============================================================================
N = 1500
SEEDS = range(5)                   # 5 réalisations pour l'auto-moyennage rapide
EPSILONS = np.linspace(0.05, 0.7, 10)  # 10 points de désordre
OMEGA_RATIOS = np.linspace(0.05, 0.7, 10) # 10 points de fréquence
DISORDER_TYPE = 'stiff'            # Désordre sur les ressorts par défaut

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_point(args):
    """Worker: compute penetration length for one (epsilon, omega_ratio, seed)."""
    eps, omega_ratio, seed = args
    chain = DisorderedChain(N=N, epsilon=eps, disorder_type=DISORDER_TYPE, seed=seed)
    result = chain.simulate(excitation='wavepacket', omega_ratio=omega_ratio,
                            steps=35000, downsample=200)
    lp = chain.extract_penetration_length(result['envelope'], result['excitation_node'])
    return eps, omega_ratio, seed, lp


def main():
    print("=" * 70)
    print("ANALYSE 05 : Longueur de Pénétration (Fréquence vs Désordre)")
    print("=" * 70)
    
    tasks = [(eps, omega_ratio, seed)
             for eps in EPSILONS
             for omega_ratio in OMEGA_RATIOS
             for seed in SEEDS]
    
    print(f"Balayage 2D : {len(EPSILONS)} epsilon x {len(OMEGA_RATIOS)} frequencies")
    print(f"Total simulations : {len(tasks)}")
    print("Calcul parallèle en cours...")
    
    with Pool() as pool:
        results = pool.map(compute_point, tasks)
    
    # Accumulate results in a 2D matrix
    # Map (eps, omega_ratio) -> list of L_p
    raw_data = {}
    for eps, omega_ratio, seed, lp in results:
        key = (eps, omega_ratio)
        if key not in raw_data:
            raw_data[key] = []
        raw_data[key].append(lp)
    
    # Compute average matrix
    Lp_matrix = np.zeros((len(EPSILONS), len(OMEGA_RATIOS)))
    for i, eps in enumerate(EPSILONS):
        for j, omega_ratio in enumerate(OMEGA_RATIOS):
            Lp_matrix[i, j] = np.mean(raw_data[(eps, omega_ratio)])
            
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Generate 2D meshgrid
    X, Y = np.meshgrid(OMEGA_RATIOS, EPSILONS)
    
    # Create contour plot
    contour = ax.contourf(X, Y, Lp_matrix, levels=15, cmap='viridis_r')
    cbar = fig.colorbar(contour, ax=ax, pad=0.03)
    cbar.set_label(r"Longueur de p\'{e}n\'{e}tration $L_p$ (sites)", fontweight='bold')
    
    # Add contour lines for clear reading
    lines = ax.contour(X, Y, Lp_matrix, levels=8, colors='black', linewidths=0.5, alpha=0.5)
    ax.clabel(lines, inline=True, fmt='%.0f', fontsize=10)
    
    ax.set_xlabel(r"Fr\'{e}quence d'excitation normalis\'{e}e $\omega / \omega_{\max}$", fontweight='bold')
    ax.set_ylabel(r"Amplitude du d\'{e}sordre $\epsilon$", fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "analyse_05_penetration_sweep.pdf")
    plt.savefig(save_path)
    print(f"--> Figure 2D sauvegardée : {save_path}")


if __name__ == '__main__':
    main()
