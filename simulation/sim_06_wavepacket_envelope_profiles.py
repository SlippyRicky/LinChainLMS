"""
sim_06_wavepacket_envelope_profiles.py
======================================
The heavy numerical engine of the pipeline. Uses Numba and Multiprocessing
to perform ultra-fast Velocity Verlet time integrations on large chains.
It computes the absolute maximum spatial envelope of a signal traveling through
the disordered chain over a long period of time.

Outputs:
    - data/wp_envelopes_eps=*_N=*.csv
"""
import numpy as np
from numba import njit
import multiprocessing as mp
import time
import os
import scipy.linalg as la

import sim_config
data_dir = sim_config.DATA_DIR

# Get parameters
params = sim_config.get_params("sim_06")
N = params['N_verlet']
M_SEEDS = params['M_realizations']
T_MAX = params['T_max']

m0, k0, a = 1.0, 1.0, 1.0
DT = 0.05
STEPS = int(T_MAX / DT)
L_ABC = 100 if N >= 200 else N // 2
gamma_max = 0.1
NODE_EXCITATION = N // 2

DISORDER_LEVELS = [0.1, 0.7, 1.4]

# Choose frequency resolution based on mode
if M_SEEDS > 2:
    freq_spacing = 150
else:
    freq_spacing = 4500  # fast test run: very few frequencies

q_modes = np.arange(1, N, 2)
OMEGAS = 2.0 * np.sqrt(k0 / m0) * np.sin(q_modes * np.pi / (2.0 * (N + 1)))
OMEGAS = OMEGAS[::freq_spacing]

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
    Velocity Verlet integrator with time-dependent central forcing.
    """
    u = np.zeros(N)
    v = np.zeros(N)
    envelope = np.zeros(N)
    
    t0 = 200.0       # Center of temporal pulse
    sigma_t = 8.0 * np.pi / omega_ext   
    sigma = 1.0 / (4.0 * sigma_t**2)
    
    accel = equation_of_motion(0.0, u, v, k_bonds, g_profile)
    
    for i in range(steps):
        t = i * dt
        t_next = t + dt
        
        # 1. Intermediate velocity
        for j in range(N):
            v[j] += 0.5 * dt * accel[j]
            
        # 2. Intermediate position
        for j in range(N):
            u[j] += v[j] * dt
            
        # 3. Forcing of the central mass
        arg_gauss = -2.0 * sigma * (t_next - t0)**2
        if arg_gauss > -50.0:
            u[NODE_EXCITATION] = np.exp(arg_gauss) * np.sin(omega_ext * (t_next - t0))
        else:
            u[NODE_EXCITATION] = 0.0
        v[NODE_EXCITATION] = 0.0
        
        # Record maximum envelope
        for j in range(N):
            if abs(u[j]) > envelope[j]: 
                envelope[j] = abs(u[j])
            
        # 4. New accelerations
        accel = equation_of_motion(t_next, u, v, k_bonds, g_profile)
        
        # 5. Final velocity
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
    print("   RUNNING SPATIAL ENVELOPES (SIM 06)                            ")
    print("=================================================================")
    print(f"Accuracy mode: {sim_config.ACCURACY_MODE}")
    print(f"N={N}, Frequencies={len(OMEGAS)}, Epsilon={DISORDER_LEVELS}")
    print("=================================================================\n")

    g_profile = np.zeros(N)
    x_abc = L_ABC * a
    for i in range(L_ABC):
        dist_left = x_abc - i * a
        g_profile[i] = gamma_max * (dist_left / x_abc)**2
        dist_right = (L_ABC - i) * a
        g_profile[N-1-i] = gamma_max * (dist_right / x_abc)**2

    num_cores = mp.cpu_count()
    print(f"-> Using {num_cores} cores.")

    for eps in DISORDER_LEVELS:
        t0 = time.time()
        print(f"\n--- Starting eps = {eps} (M={M_SEEDS} seeds) ---")

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

        LOG_FLOOR = 1e-12
        sum_log_env = np.zeros((len(omegas_exact), N))

        for seed in range(M_SEEDS):
            print(f"  -> Wavepacket seed {seed+1}/{M_SEEDS}...")
            np.random.seed(seed)
            k_bonds_s = np.ones(N+1) * k0
            if eps > 0.0:
                for i in range(N+1):
                    k_bonds_s[i] = k0 * (1.0 + eps * (np.random.random() - 0.5))

            tasks_s = [(idx, omega, k_bonds_s, g_profile) for idx, omega in enumerate(omegas_exact)]
            seed_results = np.zeros((len(omegas_exact), N))
            with mp.Pool(processes=num_cores) as pool:
                for idx, env in pool.imap_unordered(worker_wp, tasks_s):
                    seed_results[idx] = env

            center = N // 2
            for f_idx in range(len(omegas_exact)):
                row = seed_results[f_idx]
                norm = max(abs(row[center]), LOG_FLOOR)
                log_row = np.log(np.maximum(row / norm, LOG_FLOOR))
                sum_log_env[f_idx] += log_row

        avg_log_env = sum_log_env / M_SEEDS

        header = "omega_ext," + ",".join([f"u_{i}" for i in range(N)])

        # Save to the standardized file location
        out_wp = os.path.join(data_dir, f'wp_envelopes_eps={eps}_N={N}.csv')
        final_wp = np.hstack((omegas_exact.reshape(-1, 1), avg_log_env))
        np.savetxt(out_wp, final_wp, delimiter=',', header=header, comments='')

        print(f"  -> Saved CSVs for eps={eps}. Time: {time.time() - t0:.1f}s")

    print("\n=================================================================")
    print("   SIM 06 COMPLETE!                                              ")
    print("=================================================================")
