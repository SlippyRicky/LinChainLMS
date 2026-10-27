import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Choisir le niveau de désordre
    eps = 1.4
    N = 6000
    
    env_file = os.path.join(base_dir, 'Analyse', 'data', f'envelopes_eps={eps}_N={N}_test.csv')
    eig_file = os.path.join(base_dir, 'Analyse', 'data', f'eigenvectors_eps={eps}_N={N}_test.csv')
    
    if not os.path.exists(env_file) or not os.path.exists(eig_file):
        print(f"Erreur : Fichiers de données introuvables pour eps={eps}.")
        print("Générez-les d'abord avec generate_envelopes_data.py")
        return

    print("Chargement des données dynamiques et des vecteurs propres exacts...")
    data_env = np.loadtxt(env_file, delimiter=',', skiprows=1)
    data_eig = np.loadtxt(eig_file, delimiter=',', skiprows=1)
    
    # S'il n'y a qu'une seule fréquence (TEST_MODE=True), loadtxt renvoie un tableau 1D
    if data_env.ndim == 1:
        data_env = data_env.reshape(1, -1)
        data_eig = data_eig.reshape(1, -1)
        
    omegas = data_env[:, 0]
    envelopes = data_env[:, 1:]
    eigenvecs = data_eig[:, 1:]
    
    # Prendre une fréquence médiane
    mid_idx = len(omegas) // 2
    omega = omegas[mid_idx]
    
    env = envelopes[mid_idx]
    eig = np.abs(eigenvecs[mid_idx])
    
    # Extraction de la partie droite
    node_excitation = N // 2
    x_max = 2500
    x_fit = np.arange(x_max)
    
    y_env = env[node_excitation : node_excitation + x_max]
    y_eig = eig[node_excitation : node_excitation + x_max]
    
    # Normalisation pour comparer les formes
    if y_env[0] > 0: y_env /= y_env[0]
    if y_eig[0] > 0: y_eig /= y_eig[0]
    
    # Création du graphique
    plt.figure(figsize=(10, 6))
    plt.semilogy(x_fit, y_env, label="Enveloppe dynamique $u_{max}(x)$", color='blue', alpha=0.7)
    plt.semilogy(x_fit, y_eig, label="Vecteur propre exact $|e(x)|$", color='red', alpha=0.7, linestyle='--')
    
    plt.title(f"Comparaison Dynamique vs Statique - $\omega = {omega:.4f}$", fontweight='bold')
    plt.xlabel("Distance à la source $x$ (sites)", fontsize=12)
    plt.ylabel("Amplitude normalisée", fontsize=12)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim(1e-5, 2.0)
    
    # Annotation de conclusion
    plt.annotate("Si la ligne rouge s'effondre sans queue de loi de puissance,\ncela prouve que la fuite est un artefact dynamique !", 
                 xy=(0.05, 0.1), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=2, alpha=0.8),
                 fontsize=11)
    
    output_png = os.path.join(base_dir, 'scratch', f'eigenvector_comparison_eps{eps}.png')
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"Graphique de comparaison sauvegardé : {output_png}")

if __name__ == "__main__":
    main()
