"""
Analyse 06 — Balayage Fréquentiel Haute Performance avec Numba (Velocity Verlet).

Calcule l'enveloppe de déplacement pour 1500 fréquences propres et 3 désordres,
puis extrait et trace la longueur de pénétration L_p en fonction de la fréquence.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from numba import njit
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
from engine.style import apply_style, COLORS

apply_style()

# =================================================================
# PARAMÈTRES PHYSIQUES ET NUMÉRIQUES (Identiques au modèle de base)
# =================================================================
m0, k0 = 1.0, 1.0
N = 3000
DT = 0.05
STEPS = 40000       # Temps total = 2000 u.a.
L_ABC = 100
gamma_max = 0.1
NODE_EXCITATION = N // 2

# Option de Test (Mis à True par défaut pour sécurité et rapidité)
TEST_MODE = True

# Les 3 niveaux de désordre demandés
DISORDER_LEVELS = [0.1, 0.7, 1.4]

# Calcul des 1500 fréquences propres associées aux modes symétriques (impairs)
q_modes = np.arange(1, N, 2)  # Exactement 1500 modes impairs
OMEGAS = 2.0 * np.sqrt(k0 / m0) * np.sin(q_modes * np.pi / (2.0 * (N + 1)))

if TEST_MODE:
    print("--- [MODE VÉRIFICATION ACTIVÉ] ---")
    STEPS = 1000            # Simulation courte
    DISORDER_LEVELS = [0.7]  # Un seul désordre testé
    OMEGAS = OMEGAS[::300]   # Seulement 5 fréquences sur les 1500

# Répertoires de sortie
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =================================================================
# MOTEUR PHYSIQUE OPTIMISÉ (Inlined Velocity Verlet sous Numba)
# =================================================================
@njit(fastmath=True)
def run_single_simulation(omega_ext, k_bonds, g_profile, steps, dt):
    """
    Exécute une simulation unique ultra-rapide pour une fréquence et un désordre donnés.
    Ne conserve aucune histoire temporelle (u_hist, e_hist) excepté l'enveloppe finale.
    """
    u = np.zeros(N)
    v = np.zeros(N)
    a = np.zeros(N)
    envelope = np.zeros(N)
   
    # Règle de calcul dynamique et sécurisée du T_PULSE (4x la période, max 300.0)
    T_PULSE = min(8.0 * np.pi / omega_ext, 300.0)
   
    # 1. Calcul initial des accélérations à t = 0
    # Inlining de get_a pour maximiser la vitesse sous Numba
    for j in range(1, N-1):
        a[j] = -k_bonds[j]*(u[j]-u[j-1]) - k_bonds[j+1]*(u[j]-u[j+1])
    a[0] = -k_bonds[0]*u[0] - k_bonds[1]*(u[0]-u[1])
    a[N-1] = -k_bonds[N-1]*(u[N-1]-u[N-2]) - k_bonds[N]*u[N-1]
    for j in range(N):
        a[j] -= g_profile[j] * v[j]
    # Excitation à t = 0
    amp = np.exp(-(0.0 - T_PULSE)**2 / (2.0 * (T_PULSE/6.0)**2))
    a[NODE_EXCITATION] += 1.0 * amp * np.sin(0.0)
    a /= m0

    # 2. Boucle temporelle principale (Velocity Verlet)
    for i in range(steps):
        t = i * dt
        t_next = t + dt
       
        # Demi-pas en vitesse : v(n+1/2) = v(n) + dt/2 * a(n)
        for j in range(N):
            v[j] += 0.5 * dt * a[j]
           
        # Pas en position : u(n+1) = u(n) + dt * v(n+1/2)
        for j in range(N):
            u[j] += v[j] * dt
           
        # Enregistrement de l'enveloppe absolue
        for j in range(N):
            abs_u = abs(u[j])
            if abs_u > envelope[j]:
                envelope[j] = abs_u
               
        # Calcul de l'accélération suivante : a(n+1) = F(u_n+1, v_n+1/2) / m
        for j in range(1, N-1):
            a[j] = -k_bonds[j]*(u[j]-u[j-1]) - k_bonds[j+1]*(u[j]-u[j+1])
        a[0] = -k_bonds[0]*u[0] - k_bonds[1]*(u[0]-u[1])
        a[N-1] = -k_bonds[N-1]*(u[N-1]-u[N-2]) - k_bonds[N]*u[N-1]
        for j in range(N):
            a[j] -= g_profile[j] * v[j]
           
        # Excitation temporelle
        if t_next < 2.0 * T_PULSE:
            amp_next = np.exp(-(t_next - T_PULSE)**2 / (2.0 * (T_PULSE/6.0)**2))
            a[NODE_EXCITATION] += 1.0 * amp_next * np.sin(omega_ext * t_next)
           
        a /= m0
       
        # Deuxième demi-pas en vitesse : v(n+1) = v(n+1/2) + dt/2 * a(n+1)
        for j in range(N):
            v[j] += 0.5 * dt * a[j]
           
    return envelope

# =================================================================
# FONCTION WRAPPER POUR LE MULTIPROCESSING
# =================================================================
def worker(args):
    """Exécute une tâche unitaire dans le pool de processus."""
    idx, omega, k_bonds, g_profile = args
    env = run_single_simulation(omega, k_bonds, g_profile, STEPS, DT)
    return idx, env


def extract_lp(envelope):
    """Calcule la longueur de pénétration L_p depuis le nœud d'excitation central."""
    u_source = envelope[NODE_EXCITATION]
    if u_source <= 0:
        return 0.0
    target = u_source / np.exp(1.0)
    
    # On regarde vers la droite
    env_right = envelope[NODE_EXCITATION:]
    distances = np.arange(len(env_right))
    
    indices = np.where(env_right <= target)[0]
    if len(indices) == 0:
        return float(len(env_right))
        
    first_idx = indices[0]
    if first_idx == 0:
        return 0.0
        
    # Interpolation linéaire sub-site
    y1 = env_right[first_idx - 1]
    y2 = env_right[first_idx]
    x1 = distances[first_idx - 1]
    x2 = distances[first_idx]
    
    if y1 != y2:
        x_target = x1 + (target - y1) * (x2 - x1) / (y2 - y1)
    else:
        x_target = x1
    return x_target


# =================================================================
# SCRIPT DE CONTRÔLE PRINCIPAL
# =================================================================
def main():
    print("=================================================================")
    print("   DÉMARRAGE DU BALAYAGE MULTI-FRÉQUENCES HAUTE PERFORMANCE      ")
    print("=================================================================")
    print(f"Taille du système N = {N} atomes")
    print(f"Nombre de fréquences propres à tester = {len(OMEGAS)}")
    print(f"Niveaux de désordre = {DISORDER_LEVELS}")
    print(f"Pas d'intégration DT = {DT} | Nombre de pas = {STEPS} (T = {STEPS*DT} u.a.)")
    print("=================================================================\n")
   
    # 1. Génération du profil d'amortissement ABC
    g_profile = np.zeros(N)
    for i in range(L_ABC):
        g_profile[i] = gamma_max * ((L_ABC - i) / L_ABC)**2
        g_profile[N-1-i] = gamma_max * ((L_ABC - i) / L_ABC)**2
       
    # 2. Détection et préparation des cœurs du processeur
    num_cores = mp.cpu_count()
    print(f"-> Détection de {num_cores} cœurs CPU utilisables.")
   
    results_by_eps = {}
   
    # Boucle sur chaque niveau de désordre
    for eps in DISORDER_LEVELS:
        t0 = time.time()
        print(f"\n--- Lancement des simulations pour le désordre eps_k = {eps} ---")
       
        # Génération d'UN SEUL réseau désordonné pour ce niveau de désordre
        # (Garantit la rigueur physique d'un sweep sur la même chaîne physique !)
        np.random.seed(42)
        k_bonds = np.ones(N+1) * k0
        if eps > 0.0:
            for i in range(N+1):
                k_bonds[i] = k0 * (1.0 + eps * (np.random.random() - 0.5))
               
        # Préparation des tâches pour le multiprocessing
        tasks = []
        for idx, omega in enumerate(OMEGAS):
            tasks.append((idx, omega, k_bonds, g_profile))
           
        # Matrice pour stocker toutes les enveloppes
        results_matrix = np.zeros((len(OMEGAS), N))
       
        # Exécution en parallèle via un pool de processus
        print(f"Calcul en cours sur {num_cores} cœurs (Patience...)...")
        completed = 0
       
        with mp.Pool(processes=num_cores) as pool:
            for idx, env in pool.imap_unordered(worker, tasks):
                results_matrix[idx] = env
                completed += 1
                prog_step = max(1, len(OMEGAS) // 10)
                if completed % prog_step == 0:
                    print(f"   Progression : {completed}/{len(OMEGAS)} fréquences résolues ({completed/len(OMEGAS)*100:.1f}%)")
                   
        # Calculer les longueurs de pénétration pour toutes les fréquences
        lp_values = [extract_lp(results_matrix[j]) for j in range(len(OMEGAS))]
        results_by_eps[eps] = lp_values
                   
        # 3. Sauvegarde des résultats en fichier CSV dans le dossier data/
        output_data = np.hstack((OMEGAS.reshape(-1, 1), results_matrix))
       
        suffix = "_test" if TEST_MODE else ""
        filename = os.path.join(DATA_DIR, f"envelopes_eps={eps}_N={N}{suffix}.csv")
        print(f"Sauvegarde des enveloppes dans '{filename}'...")
       
        header_text = "omega_ext," + ",".join([f"u_{i}" for i in range(N)])
        np.savetxt(filename, output_data, delimiter=",", header=header_text, comments="")
       
        t1 = time.time()
        print(f"-> Désordre eps_k = {eps} TERMINÉ en {t1 - t0:.1f} secondes.")
        
    # =================================================================
    # TRACÉ ET SAUVEGARDE DU RAPPORT DE PÉNÉTRATION
    # =================================================================
    print("\nGénération de la figure comparative...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors_map = {0.1: COLORS['stiff'], 0.7: COLORS['accent1'], 1.4: COLORS['accent2']}
    
    omega_max_val = 2.0 * np.sqrt(k0 / m0)
    
    for eps in DISORDER_LEVELS:
        lbl = r"D\'esordre $\epsilon = " + f"{eps}$"
        color = colors_map.get(eps, COLORS['combined'])
        ax.plot(OMEGAS / omega_max_val, results_by_eps[eps], 'o-', label=lbl, color=color, markersize=4)
        
    ax.set_xlabel(r"Fr\'{e}quence d'excitation normalis\'{e}e $\omega / \omega_{\max}$")
    ax.set_ylabel(r"Longueur de p\'{e}n\'{e}tration $L_p$ (sites)")
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.2)
    ax.legend()
    
    plt.tight_layout()
    plot_filename = os.path.join(OUTPUT_DIR, f"analyse_06_frequency_sweep{suffix}.pdf")
    plt.savefig(plot_filename)
    print(f"--> Figure comparative sauvegardée dans : {plot_filename}")
       
    print("\n=================================================================")
    print("   BALAYAGE ET ENREGISTREMENT TERMINÉS AVEC SUCCÈS !             ")
    print("=================================================================")


if __name__ == "__main__":
    main()
