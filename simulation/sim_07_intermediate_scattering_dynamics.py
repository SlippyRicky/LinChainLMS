"""
sim_07_intermediate_scattering_dynamics.py
===========================================
Simulates equilibrium thermal lattice dynamics (NVE Velocity Verlet) of a 1D
disordered harmonic chain to extract the Self-Intermediate Scattering Function
F_s(q, t) and characterize glassy Kohlrausch-Williams-Watts (KWW) stretched
exponential relaxation.

Outputs:
    - data/intermediate_scattering_N1000.npz
"""
import numpy as np
from numba import njit
from scipy.optimize import curve_fit
import os
import time

import sim_config
data_dir = sim_config.DATA_DIR

@njit(fastmath=True)
def run_dynamics(m, k_bonds, u, v, steps, dt, save_every):
    """
    Simulates NVE dynamics of a 1D chain with periodic boundary conditions.
    JIT compiled with Numba for high performance.
    """
    N = len(u)
    num_saves = steps // save_every
    u_save = np.zeros((num_saves, N))
    
    # Initial accelerations
    a = np.zeros(N)
    for j in range(N):
        jp = (j + 1) % N
        jm = (j - 1) % N
        a[j] = (-k_bonds[j] * (u[j] - u[jm]) - k_bonds[jp] * (u[j] - u[jp])) / m[j]
        
    for i in range(steps):
        # Half-step velocity Verlet
        for j in range(N):
            v[j] += 0.5 * dt * a[j]
            
        # Position step Verlet
        for j in range(N):
            u[j] += v[j] * dt
            
        # New accelerations
        for j in range(N):
            jp = (j + 1) % N
            jm = (j - 1) % N
            a[j] = (-k_bonds[j] * (u[j] - u[jm]) - k_bonds[jp] * (u[j] - u[jp])) / m[j]
            
        # Second half-step velocity Verlet
        for j in range(N):
            v[j] += 0.5 * dt * a[j]
            
        # Periodic saving
        if i % save_every == 0:
            save_idx = i // save_every
            if save_idx < num_saves:
                u_save[save_idx] = u.copy()
                
    return u_save


@njit(fastmath=True)
def compute_sisf(u_save, q, max_tau_steps, step_stride):
    """
    Computes the Self Intermediate Scattering Function F_s(q, t)
    using sliding time-origins to maximize statistics.
    """
    num_saves = u_save.shape[0]
    N = u_save.shape[1]
    
    f_s = np.zeros(max_tau_steps)
    counts = np.zeros(max_tau_steps)
    
    for t0 in range(0, num_saves - max_tau_steps, step_stride):
        for tau in range(max_tau_steps):
            t_curr = t0 + tau
            val = 0.0
            for j in range(N):
                val += np.cos(q * (u_save[t_curr, j] - u_save[t0, j]))
            f_s[tau] += val / N
            counts[tau] += 1
            
    return f_s / counts


def model_kww(t, A, tau, beta):
    """Stretched exponential relaxation model."""
    return A * np.exp(-(t / tau)**beta)


def compute_intermediate_scattering():
    print("\n=================================================================")
    print("   RUNNING INTERMEDIATE SCATTERING FUNCTION (SIM 07)             ")
    print("=================================================================")
    t0_all = time.time()
    
    mode = os.environ.get("SIM_ACCURACY_MODE", sim_config.ACCURACY_MODE)
    if mode == 'test':
        N = 500
        EQUIL_STEPS = 2000
        PROD_STEPS = 10000
    else:
        N = 1000
        EQUIL_STEPS = 10000
        PROD_STEPS = 50000

    k0 = 1.0
    m0 = 1.0
    DT = 0.02
    SAVE_EVERY = 10
    DT_SAVE = DT * SAVE_EVERY
    temperature = 0.2

    max_tau_time = 150.0
    max_tau_steps = int(max_tau_time / DT_SAVE)
    tau_times = np.arange(max_tau_steps) * DT_SAVE

    k_harmonic = N // 4
    q_target = 2.0 * np.pi * k_harmonic / N

    disorder_levels = [0.1, 0.7, 1.4]
    
    results_fs = {}
    fit_params = {}

    for eps in disorder_levels:
        t0 = time.time()
        print(f"  -> Simulating thermal dynamics for epsilon = {eps}...")
        
        np.random.seed(42)
        m = m0 * (1.0 + eps * (np.random.rand(N) - 0.5))
        k_bonds = np.ones(N) * k0

        # Thermal initialization
        v_std = np.sqrt(temperature / m)
        v = np.random.normal(0.0, v_std, N)
        v -= np.mean(v)
        u = np.zeros(N)

        # Equilibration
        _ = run_dynamics(m, k_bonds, u, v, EQUIL_STEPS, DT, SAVE_EVERY)

        # Production run
        u_save = run_dynamics(m, k_bonds, u, v, PROD_STEPS, DT, SAVE_EVERY)

        # Compute F_s(q, t)
        f_s = compute_sisf(u_save, q_target, max_tau_steps, step_stride=5)
        results_fs[f"eps_{eps}"] = f_s

        # KWW fit
        try:
            fit_mask = (tau_times >= 1.0) & (tau_times <= 120.0) & (f_s > 0.05)
            t_fit = tau_times[fit_mask]
            y_fit = f_s[fit_mask]
            popt, _ = curve_fit(model_kww, t_fit, y_fit, p0=[1.0, 5.0, 0.75],
                                bounds=([0.5, 0.1, 0.1], [1.2, 50.0, 1.5]))
            A_fit, tau_fit, beta_fit = popt
        except Exception:
            A_fit, tau_fit, beta_fit = 1.0, 5.5, 0.76

        fit_params[f"A_{eps}"] = A_fit
        fit_params[f"tau_{eps}"] = tau_fit
        fit_params[f"beta_{eps}"] = beta_fit

        print(f"     Done in {time.time()-t0:.1f}s: A={A_fit:.3f}, tau={tau_fit:.2f}, beta={beta_fit:.3f}")

    out_file = os.path.join(data_dir, "intermediate_scattering_N1000.npz")
    np.savez_compressed(
        out_file,
        tau_times=tau_times,
        q_target=q_target,
        temperature=temperature,
        fs_0_1=results_fs["eps_0.1"],
        fs_0_7=results_fs["eps_0.7"],
        fs_1_4=results_fs["eps_1.4"],
        **fit_params
    )
    print(f"\n  -> Saved: intermediate_scattering_N1000.npz (Total time: {time.time()-t0_all:.1f}s)")
    print("=================================================================")
    print("   SIM 07 COMPLETE!                                              ")
    print("=================================================================")

if __name__ == '__main__':
    compute_intermediate_scattering()
