import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import sys

# Ajouter le répertoire parent de parametric_sweep pour importer le style
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'parametric_sweep')))
from engine.style import apply_style, COLORS

# Essayer d'appliquer le style LaTeX, sinon fallback sans LaTeX si non configuré
try:
    apply_style()
except Exception as e:
    print(f"Avertissement style : impossible d'appliquer apply_style() avec LaTeX : {e}")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "grid.alpha": 0.3
    })

# ==========================================
# --- Modèles d'Ajustement Spatiaux ---
# ==========================================
def model_exp(x, A, xi):
    """Modèle d'atténuation purement exponentiel (Anderson)"""
    return A * np.exp(-x / xi)

def model_stretch(x, A, xi, beta):
    """Modèle d'atténuation exponentiel étiré (KWW Spatial)"""
    return A * np.exp(-(x / xi)**beta)

def model_pow(x, A, alpha, x0):
    """Modèle d'atténuation en loi de puissance (Diffusif)"""
    return A * (x + x0)**(-alpha)

def calculate_r2(y_true, y_pred):
    """Calcule le coefficient de détermination R2"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

# ==========================================
# --- Traitement des Données ---
# ==========================================
def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    # Essayer de charger le fichier de production dans parametric_sweep/data
    data_file = os.path.abspath(os.path.join(base_dir, '..', 'parametric_sweep', 'data', 'envelopes_eps=0.7_N=6000.csv'))
    if not os.path.exists(data_file):
        data_file = os.path.join(base_dir, 'envelopes_eps=0.7_N=6000_test.csv')
    
    if not os.path.exists(data_file):
        print(f"Erreur : Le fichier de données '{data_file}' est introuvable.")
        print("Veuillez générer les données de production d'abord.")
        return
        
    print(f"Chargement des données depuis '{data_file}'...")
    # Lire le CSV
    data = np.loadtxt(data_file, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # Limiter le nombre de sous-graphiques pour éviter un crash si l'utilisateur calcule TOUS les modes propres.
    max_plots = 10
    if data.shape[0] > max_plots:
        step = data.shape[0] // max_plots
        data = data[::step][:max_plots]
    elif data.shape[0] > 2:
        data = data[::2]
    
    # Structure du CSV : col 0 = omega_ext, cols 1 à N = enveloppe
    omegas_exact = data[:, 0]
    envelopes = data[:, 1:]
    N = envelopes.shape[1]
    node_excitation = N // 2
    
    print(f"Nombre de fréquences chargées après sélection : {len(omegas_exact)}")
    print(f"Taille de la chaîne N = {N} atomes | Source au site {node_excitation}")
    
    # On restreint l'ajustement à la moitié droite de la chaîne (de la source vers le bord droit)
    # et on coupe avant la PML (qui commence à N - L_ABC = 6000 - 100 = 5900)
    # distance max x_max = 5900 - 3000 = 2900. On prend x_max = 2500 pour être en sécurité.
    x_max = 2500
    x_fit = np.arange(x_max)
    
    # Créer les répertoires de sortie
    os.makedirs(os.path.join(base_dir, 'scratch'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'Analyse', 'output'), exist_ok=True)
    
    # Analyse de chaque fréquence propre
    fig, axes = plt.subplots(len(omegas_exact), 1, figsize=(10, 3 * len(omegas_exact)), sharex=True)
    if len(omegas_exact) == 1:
        axes = [axes]
        
    print("\n--- Début des ajustements non-linéaires ---")
    
    for idx, (omega, env) in enumerate(zip(omegas_exact, envelopes)):
        # Extraire l'enveloppe à droite de la source
        y_raw = env[node_excitation : node_excitation + x_max]
        
        # Normalisation par l'amplitude à la source
        y_source = y_raw[0]
        if y_source <= 0:
            print(f"Fréquence {omega:.4f} : signal nul à la source, saut.")
            continue
        y_norm = y_raw / y_source
        
        # 1. Ajustement Exponentiel
        try:
            popt_exp, _ = curve_fit(model_exp, x_fit, y_norm, p0=[1.0, 100.0], bounds=((0, 1e-3), (np.inf, np.inf)), maxfev=10000)
            y_pred_exp = model_exp(x_fit, *popt_exp)
            r2_exp = calculate_r2(y_norm, y_pred_exp)
        except Exception as e:
            popt_exp, r2_exp = [np.nan, np.nan], -np.inf
            
        # 2. Ajustement Exponentiel Étiré (KWW Spatial)
        try:
            popt_str, _ = curve_fit(model_stretch, x_fit, y_norm, p0=[1.0, 100.0, 1.0], bounds=((0, 1e-3, 0.05), (np.inf, np.inf, 2.0)), maxfev=10000)
            y_pred_str = model_stretch(x_fit, *popt_str)
            r2_str = calculate_r2(y_norm, y_pred_str)
        except Exception as e:
            popt_str, r2_str = [np.nan, np.nan, np.nan], -np.inf
            
        # 3. Ajustement en Loi de Puissance (Diffusif)
        try:
            popt_pow, _ = curve_fit(model_pow, x_fit, y_norm, p0=[1.0, 1.0, 1.0], bounds=((0, 0, 1e-3), (np.inf, 10.0, np.inf)), maxfev=10000)
            y_pred_pow = model_pow(x_fit, *popt_pow)
            r2_pow = calculate_r2(y_norm, y_pred_pow)
        except Exception as e:
            popt_pow, r2_pow = [np.nan, np.nan, np.nan], -np.inf
            
        print(f"\nFréquence omega = {omega:.4f} :")
        print(f"  * Modèle Exponentiel : xi = {popt_exp[1]:.1f} | R2 = {r2_exp:.5f}")
        print(f"  * Modèle Étiré (KWW) : xi = {popt_str[1]:.1f}, beta = {popt_str[2]:.3f} | R2 = {r2_str:.5f}")
        print(f"  * Loi de Puissance   : alpha = {popt_pow[1]:.3f}, x0 = {popt_pow[2]:.1f} | R2 = {r2_pow:.5f}")
        
        # Déterminer le meilleur modèle
        r2_dict = {'Exponentiel': r2_exp, 'Étiré (KWW)': r2_str, 'Loi de Puissance': r2_pow}
        best_model = max(r2_dict, key=r2_dict.get)
        print(f"  => Meilleur modèle : {best_model} (R2 = {r2_dict[best_model]:.5f})")
        
        # --- Tracé ---
        ax = axes[idx]
        ax.plot(x_fit, y_norm, '.', label="Données de simulation", color='gray', alpha=0.5, markersize=3)
        
        if not np.isnan(popt_exp[0]):
            ax.plot(x_fit, y_pred_exp, '--', label=f"Exponentiel ($R^2={r2_exp:.4f}$, $\\xi={popt_exp[1]:.0f}$)", color=COLORS['mass'])
        if not np.isnan(popt_str[0]):
            ax.plot(x_fit, y_pred_str, '-', label=f"Étiré KWW ($R^2={r2_str:.4f}$, $\\beta={popt_str[2]:.2f}$)", color=COLORS['stiff'])
        if not np.isnan(popt_pow[0]):
            ax.plot(x_fit, y_pred_pow, ':', label=f"Loi puissance ($R^2={r2_pow:.4f}$, $\\alpha={popt_pow[1]:.2f}$)", color=COLORS['combined'])
            
        ax.set_yscale('log')
        ax.set_ylabel(r"Amplitude $|u(x)/u(0)|$")
        ax.set_ylim(1e-4, 2.0)
        ax.grid(True, which="both", alpha=0.1)
        ax.legend(loc='lower left', framealpha=0.9, fontsize=9)
        
        # Titre pour chaque panel
        ax.set_title(r"Mode propre exact $\omega = " + f"{omega:.4f}" + r"$ (D\'{e}sordre $\epsilon = 0.7$)", fontweight='bold')
        
    axes[-1].set_xlabel("Distance à la source $x$ (sites)")
    plt.tight_layout()
    
    # Sauvegarde des figures
    pdf_path_scratch = os.path.join(base_dir, 'scratch', 'spatial_envelope_fits.pdf')
    pdf_path_output = os.path.join(base_dir, 'Analyse', 'output', 'analyse_06_spatial_envelope_fits.pdf')
    
    plt.savefig(pdf_path_scratch, dpi=300)
    plt.savefig(pdf_path_output, dpi=300)
    print(f"\n--> Figure comparative sauvegardée dans : {pdf_path_scratch}")
    print(f"--> Figure comparative copiée dans : {pdf_path_output}")

if __name__ == "__main__":
    main()
