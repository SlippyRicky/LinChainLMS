import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import sys

# Ajouter le répertoire parent d'Analyse pour importer le style
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Analyse')))
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
# --- Modèles d'Ajustement Temporels ---
# ==========================================
def model_exp(t, A, tau):
    """Modèle d'atténuation purement exponentiel"""
    return A * np.exp(-t / tau)

def model_stretch(t, A, tau, beta):
    """Modèle d'atténuation exponentiel étiré (KWW Temporel)"""
    return A * np.exp(-(t / tau)**beta)

def model_pow(t, A, alpha, t0):
    """Modèle d'atténuation en loi de puissance"""
    return A * (t + t0)**(-alpha)

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
import glob
import re

def main():
    # Chemins des fichiers
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'Analyse', 'data')
    
    # Trouver tous les fichiers correspondant aux enveloppes temporelles
    data_files = glob.glob(os.path.join(data_dir, 'temporal_envelopes_*.csv'))
    
    if not data_files:
        print(f"Erreur : Aucun fichier 'temporal_envelopes_*.csv' trouvé dans '{data_dir}'.")
        print("Veuillez générer les données temporelles d'abord.")
        return
        
    # Créer les répertoires de sortie
    os.makedirs(os.path.join(base_dir, 'scratch'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'Analyse', 'output'), exist_ok=True)
    
    for data_file in sorted(data_files):
        print(f"\n{'='*50}\nTraitement de '{os.path.basename(data_file)}'...\n{'='*50}")
        
        # Extraire eps et N depuis le nom du fichier (ex: temporal_envelopes_eps=0.7_N=6000.csv)
        filename = os.path.basename(data_file)
        eps_match = re.search(r'eps=([0-9.]+)', filename)
        eps_val = eps_match.group(1) if eps_match else "Inconnu"
        n_match = re.search(r'N=([0-9]+)', filename)
        n_val = n_match.group(1) if n_match else "Inconnu"
        suffix_match = re.search(r'_N=[0-9]+(.*)\.csv', filename)
        suffix = suffix_match.group(1) if suffix_match else ""
        
        # Lire le CSV
        data = np.loadtxt(data_file, delimiter=',', skiprows=1)
        
        # S'assurer que data est bien 2D (si une seule ligne/fréquence, loadtxt renvoie un 1D)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Limiter le nombre de sous-graphiques pour éviter un crash si l'utilisateur calcule TOUS les modes propres.
        max_plots = 10
        if data.shape[0] > max_plots:
            step = data.shape[0] // max_plots
            data = data[::step][:max_plots]
        elif data.shape[0] > 2:
            data = data[::2]
        
        # Structure du CSV : col 0 = omega_ext, cols 1 à num_time_steps = enveloppe temporelle
        omegas_exact = data[:, 0]
        envelopes = data[:, 1:]
        num_time_steps = envelopes.shape[1]
        
        print(f"Nombre de fréquences chargées après sélection : {len(omegas_exact)}")
        print(f"Nombre de pas de temps enregistrés = {num_time_steps}")
        
        # On ajuste sur la longueur des données temporelles.
        # Modifier t_max ou l'indice de début (t_start) selon la structure exacte des données
        t_start = 0
        t_max = min(2500, num_time_steps) # Ajuster la borne si besoin
        t_fit = np.arange(t_max)
        
        # Analyse de chaque fréquence propre
        fig, axes = plt.subplots(len(omegas_exact), 1, figsize=(10, 3 * len(omegas_exact)), sharex=True)
        if len(omegas_exact) == 1:
            axes = [axes]
            
        print("\n--- Début des ajustements non-linéaires ---")
        
        for idx, (omega, env) in enumerate(zip(omegas_exact, envelopes)):
            # Extraire l'enveloppe temporelle (de t_start à t_start + t_max)
            y_raw = env[t_start : t_start + t_max]
            
            # Normalisation par l'amplitude maximale
            y_source = np.max(y_raw)
            if y_source <= 0:
                print(f"Fréquence {omega:.4f} : signal nul partout, saut.")
                continue
            y_norm = y_raw / y_source
            
            # 1. Ajustement Exponentiel
            try:
                popt_exp, _ = curve_fit(model_exp, t_fit, y_norm, p0=[1.0, 100.0], bounds=((0, 1e-3), (np.inf, np.inf)), maxfev=10000)
                y_pred_exp = model_exp(t_fit, *popt_exp)
                r2_exp = calculate_r2(y_norm, y_pred_exp)
            except Exception as e:
                popt_exp, r2_exp = [np.nan, np.nan], -np.inf
                
            # 2. Ajustement Exponentiel Étiré (KWW Temporel)
            try:
                popt_str, _ = curve_fit(model_stretch, t_fit, y_norm, p0=[1.0, 100.0, 1.0], bounds=((0, 1e-3, 0.05), (np.inf, np.inf, 2.0)), maxfev=10000)
                y_pred_str = model_stretch(t_fit, *popt_str)
                r2_str = calculate_r2(y_norm, y_pred_str)
            except Exception as e:
                popt_str, r2_str = [np.nan, np.nan, np.nan], -np.inf
                
            # 3. Ajustement en Loi de Puissance
            try:
                popt_pow, _ = curve_fit(model_pow, t_fit, y_norm, p0=[1.0, 1.0, 1.0], bounds=((0, 0, 1e-3), (np.inf, 10.0, np.inf)), maxfev=10000)
                y_pred_pow = model_pow(t_fit, *popt_pow)
                r2_pow = calculate_r2(y_norm, y_pred_pow)
            except Exception as e:
                popt_pow, r2_pow = [np.nan, np.nan, np.nan], -np.inf
                
            print(f"\nFréquence omega = {omega:.4f} :")
            print(f"  * Modèle Exponentiel : tau = {popt_exp[1]:.1f} | R2 = {r2_exp:.5f}")
            print(f"  * Modèle Étiré (KWW) : tau = {popt_str[1]:.1f}, beta = {popt_str[2]:.3f} | R2 = {r2_str:.5f}")
            print(f"  * Loi de Puissance   : alpha = {popt_pow[1]:.3f}, t0 = {popt_pow[2]:.1f} | R2 = {r2_pow:.5f}")
            
            # Déterminer le meilleur modèle
            r2_dict = {'Exponentiel': r2_exp, 'Étiré (KWW)': r2_str, 'Loi de Puissance': r2_pow}
            best_model = max(r2_dict, key=r2_dict.get)
            print(f"  => Meilleur modèle : {best_model} (R2 = {r2_dict[best_model]:.5f})")
            
            # --- Tracé ---
            ax = axes[idx]
            ax.plot(t_fit, y_norm, '.', label="Données temporelles", color='gray', alpha=0.5, markersize=3)
            
            if not np.isnan(popt_exp[0]):
                ax.plot(t_fit, y_pred_exp, '--', label=f"Exponentiel ($R^2={r2_exp:.4f}$, $\\tau={popt_exp[1]:.0f}$)", color=COLORS['mass'])
            if not np.isnan(popt_str[0]):
                ax.plot(t_fit, y_pred_str, '-', label=f"Étiré KWW ($R^2={r2_str:.4f}$, $\\beta={popt_str[2]:.2f}$)", color=COLORS['stiff'])
            if not np.isnan(popt_pow[0]):
                ax.plot(t_fit, y_pred_pow, ':', label=f"Loi puissance ($R^2={r2_pow:.4f}$, $\\alpha={popt_pow[1]:.2f}$)", color=COLORS['combined'])
                
            ax.set_yscale('log')
            ax.set_ylabel(r"Amplitude $|u(t)/u(0)|$")
            ax.set_ylim(1e-4, 2.0)
            ax.grid(True, which="both", alpha=0.1)
            ax.legend(loc='lower left', framealpha=0.9, fontsize=9)
            
            # Titre pour chaque panel
            ax.set_title(r"Mode propre exact $\omega = " + f"{omega:.4f}" + rf"$ (D\'{{e}}sordre $\epsilon = {eps_val}$)", fontweight='bold')
            
        axes[-1].set_xlabel("Temps $t$ (u.a.)")
        plt.tight_layout()
        
        # Sauvegarde des figures (avec le désordre et le type dans le nom de fichier)
        file_suffix = f"eps={eps_val}_N={n_val}{suffix}"
        pdf_path_scratch = os.path.join(base_dir, 'scratch', f'temporal_envelope_fits_{file_suffix}.pdf')
        pdf_path_output = os.path.join(base_dir, 'Analyse', 'output', f'analyse_06_temporal_envelope_fits_{file_suffix}.pdf')
        
        plt.savefig(pdf_path_scratch, dpi=300)
        plt.savefig(pdf_path_output, dpi=300)
        plt.close(fig) # Fermer la figure pour libérer la mémoire pour la prochaine boucle
        print(f"\n--> Figure temporelle sauvegardée dans : {pdf_path_scratch}")
        print(f"--> Figure temporelle copiée dans : {pdf_path_output}")

if __name__ == "__main__":
    main()
