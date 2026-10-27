import numpy as np
from numba import njit, prange
import multiprocessing as mp
import time
import os
import scipy.linalg as la

# =================================================================
# PARAMÈTRES PHYSIQUES ET NUMÉRIQUES (Identiques au modèle de base)
# =================================================================
m0, k0 = 1.0, 1.0
N = 6000
DT = 0.05
STEPS = 80000       # Temps total = 4000 u.a.
DOWNSAMPLE = 10     # Enregistrer 1 point sur 10 pour le profil temporel
L_ABC = 100
gamma_max = 0.1
NODE_EXCITATION = N // 2

# Option de Test
TEST_MODE = True

# Les 3 niveaux de désordre demandés
DISORDER_LEVELS = [0.1, 0.7, 1.4]

q_modes = np.arange(1, N, 2)  # Exactement N/2 modes impairs
OMEGAS = 2.0 * np.sqrt(k0 / m0) * np.sin(q_modes * np.pi / (2.0 * (N + 1)))
OMEGAS = OMEGAS[::150]       # Downsample to 10 representative frequencies

if TEST_MODE:
    print("--- [MODE VÉRIFICATION ACTIVÉ] ---")
    STEPS = 1000            # Simulation courte
    DISORDER_LEVELS = [0.1, 0.7, 1.4]  # Les désordres
    OMEGAS = OMEGAS[::300]   # Seulement 5 fréquences sur les 1500


# =================================================================
# MOTEUR PHYSIQUE OPTIMISÉ (Inlined Velocity Verlet sous Numba)
# =================================================================
@njit(fastmath=True)
def run_single_simulation(omega_ext, k_bonds, g_profile, steps, dt):
    """
    Exécute une simulation unique ultra-rapide pour une fréquence et un désordre donnés.
    Conserve le maximum spatial de |u| au cours du temps (l'enveloppe temporelle).
    """
    u = np.zeros(N)
    v = np.zeros(N)
    a = np.zeros(N)
    
    n_records = steps // DOWNSAMPLE
    if steps % DOWNSAMPLE != 0:
        n_records += 1
    temporal_envelope = np.zeros(n_records)

    T_PULSE = min(8.0 * np.pi / omega_ext, 300.0)

    # 1. Calcul initial des accélérations à t = 0
    for j in range(1, N-1):
        a[j] = -k_bonds[j]*(u[j]-u[j-1]) - k_bonds[j+1]*(u[j]-u[j+1])
    a[0] = -k_bonds[0]*u[0] - k_bonds[1]*(u[0]-u[1])
    a[N-1] = -k_bonds[N-1]*(u[N-1]-u[N-2]) - k_bonds[N]*u[N-1]
    for j in range(N):
        a[j] -= g_profile[j] * v[j]
        
    amp = np.exp(-(0.0 - T_PULSE)**2 / (2.0 * (T_PULSE/6.0)**2))
    a[NODE_EXCITATION] += 1.0 * amp * np.sin(0.0)
    a /= m0

    record_idx = 0
    # 2. Boucle temporelle principale (Velocity Verlet)
    for i in range(steps):
        t = i * dt
        t_next = t + dt

        # Demi-pas en vitesse : v(n+1/2)
        for j in range(N):
            v[j] += 0.5 * dt * a[j]

        # Pas en position : u(n+1)
        for j in range(N):
            u[j] += v[j] * dt

        # Enregistrement de l'enveloppe temporelle : max(|u(x,t)|) sur l'espace
        if i % DOWNSAMPLE == 0:
            max_u = 0.0
            for j in range(N):
                abs_u = abs(u[j])
                if abs_u > max_u:
                    max_u = abs_u
            temporal_envelope[record_idx] = max_u
            record_idx += 1

        # Calcul de l'accélération suivante
        for j in range(1, N-1):
            a[j] = -k_bonds[j]*(u[j]-u[j-1]) - k_bonds[j+1]*(u[j]-u[j+1])
        a[0] = -k_bonds[0]*u[0] - k_bonds[1]*(u[0]-u[1])
        a[N-1] = -k_bonds[N-1]*(u[N-1]-u[N-2]) - k_bonds[N]*u[N-1]
        for j in range(N):
            a[j] -= g_profile[j] * v[j]

        if t_next < 2.0 * T_PULSE:
            amp_next = np.exp(-(t_next - T_PULSE)**2 / (2.0 * (T_PULSE/6.0)**2))
            a[NODE_EXCITATION] += 1.0 * amp_next * np.sin(omega_ext * t_next)

        a /= m0

        # Deuxième demi-pas en vitesse : v(n+1)
        for j in range(N):
            v[j] += 0.5 * dt * a[j]

    return temporal_envelope

# =================================================================
# FONCTION WRAPPER POUR LE MULTIPROCESSING
# =================================================================
def worker(args):
    idx, omega, k_bonds, g_profile = args
    env = run_single_simulation(omega, k_bonds, g_profile, STEPS, DT)
    return idx, env

# =================================================================
# SCRIPT DE CONTRÔLE PRINCIPAL
# =================================================================
if __name__ == "__main__":
    print("=================================================================")
    print("   GÉNÉRATION DES ENVELOPPES TEMPORELLES HAUTE PERFORMANCE       ")
    print("=================================================================")
    print(f"Taille du système N = {N} atomes")
    print(f"Nombre de fréquences propres à tester = {len(OMEGAS)}")
    print(f"Niveaux de désordre = {DISORDER_LEVELS}")
    print(f"Pas d'intégration DT = {DT} | Nombre de pas = {STEPS} (T = {STEPS*DT} u.a.)")
    print(f"Downsampling temporel = {DOWNSAMPLE} (1 point tous les {DOWNSAMPLE*DT} u.a.)")
    print("=================================================================\n")

    g_profile = np.zeros(N)
    for i in range(L_ABC):
        g_profile[i] = gamma_max * ((L_ABC - i) / L_ABC)**2
        g_profile[N-1-i] = gamma_max * ((L_ABC - i) / L_ABC)**2

    num_cores = mp.cpu_count()
    print(f"-> Détection de {num_cores} cœurs CPU utilisables.")

    for eps in DISORDER_LEVELS:
        t0 = time.time()
        print(f"\n--- Lancement des simulations temporelles pour eps_k = {eps} ---")

        np.random.seed(42)
        k_bonds = np.ones(N+1) * k0
        if eps > 0.0:
            for i in range(N+1):
                k_bonds[i] = k0 * (1.0 + eps * (np.random.random() - 0.5))

        main_diag = np.zeros(N)
        off_diag = np.zeros(N-1)
        main_diag[0] = k_bonds[0] / m0
        main_diag[-1] = k_bonds[-1] / m0
        for i in range(1, N-1):
            main_diag[i] = (k_bonds[i-1] + k_bonds[i]) / m0
        for i in range(N-1):
            off_diag[i] = -k_bonds[i] / m0

        eigvals, eigvecs = la.eigh_tridiagonal(main_diag, off_diag, eigvals_only=False)
        eigvals = np.maximum(eigvals, 0)
        omegas_all = np.sqrt(eigvals)
        
        omegas_pure = 2.0 * np.sqrt(k0 / m0) * np.sin(np.arange(1, N+1) * np.pi / (2.0 * (N + 1)))
        indices = [np.argmin(np.abs(omegas_pure - o)) for o in OMEGAS]
        omegas_exact = omegas_all[indices]

        tasks = []
        for idx, omega in enumerate(omegas_exact):
            tasks.append((idx, omega, k_bonds, g_profile))

        # La taille d'une enveloppe temporelle
        n_records = STEPS // DOWNSAMPLE
        if STEPS % DOWNSAMPLE != 0:
            n_records += 1

        results_matrix = np.zeros((len(omegas_exact), n_records))

        print(f"Calcul en cours sur {num_cores} cœurs (Patience...)...")
        completed = 0

        with mp.Pool(processes=num_cores) as pool:
            for idx, env in pool.imap_unordered(worker, tasks):
                results_matrix[idx] = env
                completed += 1
                prog_step = max(1, len(omegas_exact) // 10)
                if completed % prog_step == 0:
                    print(f"   Progression : {completed}/{len(omegas_exact)} fréquences résolues ({completed/len(omegas_exact)*100:.1f}%)")

        suffix = "_test" if TEST_MODE else ""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        os.makedirs(os.path.join(base_dir, 'Analyse', 'data'), exist_ok=True)
        
        output_file = os.path.join(base_dir, 'Analyse', 'data', f'temporal_envelopes_eps={eps}_N={N}{suffix}.csv')
        
        final_data = np.hstack((omegas_exact.reshape(-1, 1), results_matrix))
        
        header = "omega_ext," + ",".join([f"t_{i}" for i in range(n_records)])
        np.savetxt(output_file, final_data, delimiter=',', header=header, comments='')
        print(f"[{eps}] -> Fichier d'enveloppes temporelles sauvegardé : {os.path.basename(output_file)}")
        
        t1 = time.time()
        print(f"-> Désordre eps_k = {eps} TERMINÉ en {t1 - t0:.1f} secondes.")

    print("\n=================================================================")
    print("   GÉNÉRATION DES FICHIERS CSV TEMPORELS TERMINÉE !              ")
    print("=================================================================")
