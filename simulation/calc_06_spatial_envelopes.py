"""
calc_06_spatial_envelopes.py
============================
The heavy numerical engine of the pipeline. Uses Numba and Multiprocessing
to perform ultra-fast Velocity Verlet time integrations on large chains (N=6000).
It computes the absolute maximum spatial envelope of a signal traveling through
the disordered chain over a long period of time.

It performs two parallel sweeps across all relevant frequencies:
1. Continuous Harmonic Excitation (Green's function source at center)
2. Gaussian Wavepacket Initial Condition (Centered wavepacket)

Outputs:
    - data/envelopes_eps=*_N=6000.csv
    - data/wp_envelopes_eps=*_N=6000.csv
    - data/eigenvectors_eps=*_N=6000.csv
"""
import numpy as np
from numba import njit
import multiprocessing as mp
import time
import os
import scipy.linalg as la

# =================================================================
# PARAMÈTRES PHYSIQUES ET NUMÉRIQUES
# =================================================================

TEST_MODE = False

m0, k0, a = 1.0, 1.0, 1.0

N = 6000
DT = 0.05
STEPS = 80000       
L_ABC = 100
gamma_max = 0.1
NODE_EXCITATION = N // 2

DISORDER_LEVELS = [0.1, 0.7, 1.4]
M_SEEDS = 10  # Number of disorder realizations to average over

q_modes = np.arange(1, N, 2)
OMEGAS = 2.0 * np.sqrt(k0 / m0) * np.sin(q_modes * np.pi / (2.0 * (N + 1)))
OMEGAS = OMEGAS[::150]

if TEST_MODE:
    STEPS = 1000
    OMEGAS = OMEGAS[::300]

import sim_config
data_dir = sim_config.DATA_DIR

# =================================================================
# Numba Engines
# =================================================================
@njit(fastmath=True)
def equation_of_motion(t, u, v, k_bonds, g_profile):
    """
    Computes the acceleration of the chain at time t.
    d^2 u_j / dt^2 = (-k_j*(u_j - u_{j-1}) - k_{j+1}*(u_j - u_{j+1}) - g_profile[j]*v_j) / m0
    """
    accel = np.zeros(N)
    
    # Boundary masses
    accel[0] = -k_bonds[0]*u[0] - k_bonds[1]*(u[0]-u[1])
    accel[N-1] = -k_bonds[N-1]*(u[N-1]-u[N-2]) - k_bonds[N]*u[N-1]
    
    # Interior masses
    for j in range(1, N-1):
        accel[j] = -k_bonds[j]*(u[j]-u[j-1]) - k_bonds[j+1]*(u[j]-u[j+1])
        
    # Damping and division by mass m0
    for j in range(N):
        accel[j] = (accel[j] - g_profile[j] * v[j]) / m0
        
    return accel


@njit(fastmath=True)
def run_wp_simulation(omega_ext, k_bonds, g_profile, steps, dt):
    """
    Intégrateur Velocity Verlet avec forçage temporel sur la masse centrale (N/2).
    """
    u = np.zeros(N)
    v = np.zeros(N)
    envelope = np.zeros(N)
    
    # Paramètres de l'excitation temporelle
    t0 = 200.0       # Centre de l'impulsion temporelle
    # La largeur de l'impulsion est choisie proportionnelle à la période d'excitation
    # Cela permet de maintenir une erreur relative constante sur la largeur de fréquence (sigma(t) = 8*pi / omega_ext)
    sigma_t = 8.0 * np.pi / omega_ext   
    sigma = 1.0 / (4.0 * sigma_t**2) # Convertie pour la formule exp(-2*sigma*t^2)
    
    # Initialisation à t=0
    accel = equation_of_motion(0.0, u, v, k_bonds, g_profile)
    
    for i in range(steps):
        t = i * dt
        t_next = t + dt
        
        # 1. Vitesse intermédiaire
        for j in range(N):
            v[j] += 0.5 * dt * accel[j]
            
        # 2. Position intermédiaire
        for j in range(N):
            u[j] += v[j] * dt
            
        # 3. FORÇAGE de la masse centrale (NODE_EXCITATION = N // 2)
        # Formule demandée : exp(-2 * sigma * (t - t0)^2) * sin(omega * (t - t0))
        arg_gauss = -2.0 * sigma * (t_next - t0)**2
        # Sécurité numérique pour l'exponentielle
        if arg_gauss > -50.0:
            u[NODE_EXCITATION] = np.exp(arg_gauss) * np.sin(omega_ext * (t_next - t0))
        else:
            u[NODE_EXCITATION] = 0.0
        v[NODE_EXCITATION] = 0.0  # On fixe sa vitesse puisque son mouvement est imposé
        
        # Enregistrement de l'enveloppe maximale atteinte par chaque masse
        for j in range(N):
            if abs(u[j]) > envelope[j]: envelope[j] = abs(u[j])
            
        # 4. Nouvelles accélérations
        accel = equation_of_motion(t_next, u, v, k_bonds, g_profile)
        
        # 5. Vitesse finale
        for j in range(N):
            v[j] += 0.5 * dt * accel[j]
            
    return envelope


def worker_wp(args):
    """Multiprocessing wrapper for run_wp_simulation."""
    idx, omega, k_bonds, g_profile = args
    env = run_wp_simulation(omega, k_bonds, g_profile, STEPS, DT)
    return idx, env



if __name__ == "__main__":
    print("=================================================================")
    print("   RUNNING SPATIAL ENVELOPES (CALC 06)                           ")
    print("=================================================================")
    if TEST_MODE: print("   [TEST MODE ENABLED - FAST RUN]                        ")
    print(f"N={N}, Frequencies={len(OMEGAS)}, Epsilon={DISORDER_LEVELS}")
    print("=================================================================\n")

    g_profile = np.zeros(N)
    x_abc = L_ABC * a
    for i in range(L_ABC):
        # Left boundary: position x_pos = i * a, distance from PML start is x_abc - x_pos
        dist_left = x_abc - i * a
        g_profile[i] = gamma_max * (dist_left / x_abc)**2
        # Right boundary: position x_pos = (N - 1 - i) * a, distance from PML start is (L_ABC - i) * a
        dist_right = (L_ABC - i) * a
        g_profile[N-1-i] = gamma_max * (dist_right / x_abc)**2

    num_cores = mp.cpu_count()
    print(f"-> Using {num_cores} cores.")

    for eps in DISORDER_LEVELS:
        t0 = time.time()
        print(f"\n--- Starting eps = {eps} (M={M_SEEDS} seeds) ---")

        # We need one representative realization to get the exact modes of the chain.
        # Use seed 42 (matching reference seed).
        np.random.seed(42)
        k_bonds_ref = np.ones(N+1) * k0
        if eps > 0.0:
            for i in range(N+1):
                k_bonds_ref[i] = k0 * (1.0 + eps * (np.random.random() - 0.5))

        main_diag = np.zeros(N)
        off_diag = np.zeros(N-1)
        main_diag[0] = k_bonds_ref[0] / m0
        main_diag[-1] = k_bonds_ref[-1] / m0
        for i in range(1, N-1):
            main_diag[i] = (k_bonds_ref[i-1] + k_bonds_ref[i]) / m0
        for i in range(N-1):
            off_diag[i] = -k_bonds_ref[i] / m0

        eigvals = la.eigvalsh_tridiagonal(main_diag, off_diag)
        eigvals = np.maximum(eigvals, 0)
        omegas_all = np.sqrt(eigvals)

        omegas_pure = 2.0 * np.sqrt(k0 / m0) * np.sin(np.arange(1, N+1) * np.pi / (2.0 * (N + 1)))
        indices = [np.argmin(np.abs(omegas_pure - o)) for o in OMEGAS]
        omegas_exact = omegas_all[indices]

        # --- RUN WAVEPACKET (M_SEEDS seeds, ensemble-average log-envelope) ---
        # We store sum_of_log_envelopes and divide by M_SEEDS at the end.
        # log-envelope at position n = log(|u_n| / |u_center|), averaged over seeds.
        # A small floor (1e-12) prevents log(0).
        LOG_FLOOR = 1e-12
        sum_log_env = np.zeros((len(omegas_exact), N))

        for seed in range(M_SEEDS):
            print(f"  -> Wavepacket seed {seed+1}/{M_SEEDS}...")
            np.random.seed(seed)  # each seed gives a different disorder
            k_bonds_s = np.ones(N+1) * k0
            if eps > 0.0:
                for i in range(N+1):
                    k_bonds_s[i] = k0 * (1.0 + eps * (np.random.random() - 0.5))

            tasks_s = [(idx, omega, k_bonds_s, g_profile) for idx, omega in enumerate(omegas_exact)]
            seed_results = np.zeros((len(omegas_exact), N))
            with mp.Pool(processes=num_cores) as pool:
                for idx, env in pool.imap_unordered(worker_wp, tasks_s):
                    seed_results[idx] = env

            # Normalize each row by its center value, then take log
            center = N // 2
            for f_idx in range(len(omegas_exact)):
                row = seed_results[f_idx]
                norm = max(abs(row[center]), LOG_FLOOR)
                log_row = np.log(np.maximum(row / norm, LOG_FLOOR))
                sum_log_env[f_idx] += log_row

        # Ensemble-averaged log-envelope
        avg_log_env = sum_log_env / M_SEEDS

        # --- SAVING ---
        suffix = "_test" if TEST_MODE else ""
        header = "omega_ext," + ",".join([f"u_{i}" for i in range(N)])

        # Wavepacket: save the ensemble-averaged log-envelope
        out_wp = os.path.join(data_dir, f'wp_envelopes_eps={eps}_N={N}{suffix}.csv')
        final_wp = np.hstack((omegas_exact.reshape(-1, 1), avg_log_env))
        np.savetxt(out_wp, final_wp, delimiter=',', header=header, comments='')

        print(f"  -> Saved CSVs for eps={eps}. Time: {time.time() - t0:.1f}s")

    print("\n=================================================================")
    print("   CALC 06 COMPLETE!                                             ")
    print("=================================================================")
