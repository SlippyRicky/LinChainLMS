import numpy as np
import matplotlib.pyplot as plt
import os

def find_data_file(eps, N, test=False):
    """Recherche le fichier de données dans plusieurs répertoires possibles."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    suffix = "_test" if test else ""
    filename = f"envelopes_eps={eps}_N={N}{suffix}.csv"
    
    paths_to_check = [
        os.path.join(project_root, "code", "parametric_sweep", "data", filename),
        os.path.join(project_root, "code", "scratch", filename),
        os.path.join(project_root, "code", "scratch", "Analyse", "data", filename),
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "Analyse", "data", filename)
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            return path
            
    return None

def plot_disorder_envelopes(eps, N):
    # Recherche du fichier de production, puis du fichier test si nécessaire
    data_file = find_data_file(eps, N, test=False)
    if data_file is None:
        data_file = find_data_file(eps, N, test=True)
        
    if data_file is None:
        print(f"Avertissement : Fichier de données introuvable pour eps={eps}, N={N}")
        return False
        
    print(f"Lecture du fichier {os.path.basename(data_file)}...")
    try:
        data = np.loadtxt(data_file, delimiter=',', skiprows=1)
    except Exception as e:
        print(f"Erreur de lecture : {e}")
        return False
        
    print(f"Taille des données chargées : {data.shape[0]} lignes x {data.shape[1]} colonnes.")
    
    # Configuration esthétique pour les tracés
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "lines.linewidth": 1.0,
        "grid.alpha": 0.3
    })
    
    # 1. Sélectionner et aligner les données
    # Le premier élément de chaque ligne est omega_ext.
    # Les colonnes suivantes de 1 à N correspondent à u_0 à u_N-1.
    # Le nœud d'excitation n_c est à N // 2 = 3000 (Python index 3000, soit la colonne 3001).
    # La moitié droite de la chaîne va de u_3000 à u_5999 (les colonnes 3001 à 6000).
    col_start = 1 + N // 2
    col_end = col_start + N // 2
    
    # Extraction des enveloppes (on échantillonne une courbe sur 100 pour la lisibilité)
    step = 100
    envelopes = data[1::step, col_start:col_end]
    
    # Si le fichier test contient trop peu de lignes pour le pas de 100, on prend tout
    if envelopes.shape[0] == 0:
        envelopes = data[:, col_start:col_end]
        
    # Y a pour dimensions : (nombre de points d'espace) x (nombre de fréquences échantillonnées)
    Y = envelopes.T
    
    # L'axe X est la distance par rapport à l'excitation (X = 0 correspond exactement au nœud d'excitation)
    X = np.arange(0, N // 2)
    
    # 2. Création des tracés multi-échelles
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Enveloppes spatiales - Désordre $\\epsilon = {eps}$ (N = {N})", fontsize=16)
    
    # --- Tracé 1 : Linéaire / Linéaire ---
    ax = axes[0]
    ax.plot(X, Y)
    ax.grid(True)
    ax.set_title("Échelle Linéaire / Linéaire")
    ax.set_xlabel("Distance depuis le nœud d'excitation $x$")
    ax.set_ylabel("Amplitude de l'enveloppe $|u(x)|$")
    
    # --- Tracé 2 : Linéaire / Log ---
    ax = axes[1]
    ax.semilogy(X, Y)
    ax.grid(True, which="both")
    ax.set_title("Échelle Linéaire / Log")
    ax.set_xlabel("Distance depuis le nœud d'excitation $x$")
    ax.set_ylabel("Amplitude $|u(x)|$ (Log)")
    ax.set_ylim(1e-5, max(1.0, np.max(Y) * 1.1))
    
    # --- Tracé 3 : Log / Log ---
    # Pour éviter log(0), on commence à X = 1 (distance de 1 site par rapport à l'excitation)
    ax = axes[2]
    ax.loglog(X[1:], Y[1:, :])
    ax.grid(True, which="both")
    ax.set_title("Échelle Log / Log")
    ax.set_xlabel("Distance depuis le nœud d'excitation $x$ (Log)")
    ax.set_ylabel("Amplitude $|u(x)|$ (Log)")
    ax.set_ylim(1e-5, max(1.0, np.max(Y) * 1.1))
    
    plt.tight_layout()
    
    # Sauvegarde de la figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_png = os.path.join(script_dir, f"envelopes_eps={eps}_N={N}.png")
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    print(f"Graphiques générés pour eps={eps} : {output_png}")
    return True

def main():
    disorder_levels = [0.1, 0.7, 1.4]
    N = 6000
    
    for eps in disorder_levels:
        plot_disorder_envelopes(eps, N)

if __name__ == "__main__":
    main()
