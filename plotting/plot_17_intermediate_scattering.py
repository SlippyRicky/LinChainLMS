import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# pyrefly: ignore [missing-import]
from numba import njit
import os
import sys

# --- STYLE CONFIGURATION (Premium Manuscript Style) ---
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 21,
    "legend.fontsize": 17,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.2
})
COLORS = {'ordered': '#1f77b4', 'accent1': '#ff7f0e', 'stiff': '#2ca02c'}

# ==========================================
# --- JIT OPTIMIZED PHYSICS ENGINE ---
# ==========================================
@njit(fastmath=True)
def run_dynamics(m, k_bonds, u, v, steps, dt, save_every):
    """
    Simulates NVE dynamics of a 1D chain with periodic boundary conditions.
    JIT compiled with Numba for performance.
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

# ==========================================
# --- Kohlrausch-Williams-Watts model ---
# ==========================================
def model_kww(t, A, tau, beta):
    """Stretched exponential relaxation model"""
    return A * np.exp(-(t / tau)**beta)

# ==========================================
# --- Main Figure Plotting Execution ---
# ==========================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import plot_config
    fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
    # Directory creation handled by plot_config
    
    # Physical and Simulation Parameters
    N = 1000                # Number of sites
    k0 = 1.0                # Stiffness
    m0 = 1.0                # Nominal mass
    DT = 0.02               # Time step
    EQUIL_STEPS = 10000     # Thermal equilibration
    PROD_STEPS = 50000      # Production trajectory
    SAVE_EVERY = 10         # Save interval
    DT_SAVE = DT * SAVE_EVERY # Saving time interval = 0.2
    
    temperature = 0.2
    
    # Analysis Parameters
    max_tau_time = 150.0    # Max tau time
    max_tau_steps = int(max_tau_time / DT_SAVE)
    
    # target wavenumber (Brillouin zone middle, q*a = pi/2)
    k_harmonic = N // 4
    q_target = 2.0 * np.pi * k_harmonic / N
    
    print(f"Simulating thermal chain of N = {N} sites at T = {temperature}")
    print(f"Wavenumber q = {q_target:.4f} rad/a")
    
    disorder_levels = [0.1, 0.7, 1.4]
    
    # Create 2-subplot figure (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(11.76, 4.91))
    ax_a = axes[0]
    ax_b = axes[1]
    
    colors_map = {0.1: COLORS['ordered'], 0.7: COLORS['accent1'], 1.4: COLORS['stiff']}
    
    for eps in disorder_levels:
        print(f"\n--- Running dynamic simulation for epsilon = {eps} ---")
        
        # 1. Generate stochastic disorder (mass fluctuations)
        np.random.seed(42)
        m = m0 * (1.0 + eps * (np.random.rand(N) - 0.5))
        k_bonds = np.ones(N) * k0
        
        # 2. Boltzmann thermal initialization
        v = np.random.normal(0.0, np.sqrt(temperature / m), N)
        u = np.random.normal(0.0, np.sqrt(temperature / k0), N)
        
        # Eliminate center-of-mass velocity
        v_com = np.sum(m * v) / np.sum(m)
        v -= v_com
        
        # 3. Equilibration phase (NVE ensemble)
        print("   Equilibrating...")
        u_save_eq = run_dynamics(m, k_bonds, u, v, EQUIL_STEPS, DT, SAVE_EVERY)
        u_init = u_save_eq[-1].copy()
        
        # 4. Production phase
        print("   Production trajectory...")
        u_traj = run_dynamics(m, k_bonds, u_init, v, PROD_STEPS, DT, SAVE_EVERY)
        
        # 5. Compute F_s(q, t)
        print("   Computing Self Intermediate Scattering Function...")
        f_s = compute_sisf(u_traj, q_target, max_tau_steps, 50)
        
        t_fit = np.arange(max_tau_steps) * DT_SAVE
        
        # 6. Fit stretched exponential (KWW)
        fit_idx = np.where(f_s > 0.05)[0]
        A_kww = 1.0
        tau_kww = 10.0
        beta_kww = 1.0
        if len(fit_idx) > 10:
            try:
                popt, _ = curve_fit(model_kww, t_fit[fit_idx], f_s[fit_idx], 
                                    p0=[1.0, 10.0, 1.0], 
                                    bounds=((0.8, 1e-1, 0.05), (1.2, 500.0, 2.0)), 
                                    maxfev=10000)
                A_kww, tau_kww, beta_kww = popt
                print(f"   => Fit KWW: tau = {tau_kww:.2f} | beta = {beta_kww:.3f}")
                
                # Plot KWW fits on both subplots
                t_plot_a = np.linspace(0, max_tau_time, 300)
                y_fit_a = model_kww(t_plot_a, *popt)
                ax_a.plot(t_plot_a, y_fit_a, '--', color=colors_map[eps], alpha=0.8)
                
                t_plot_b = np.logspace(np.log10(0.2), np.log10(max_tau_time), 300)
                y_fit_b = -np.log(model_kww(t_plot_b, *popt) / A_kww)
                ax_b.plot(t_plot_b, y_fit_b, '--', color=colors_map[eps], alpha=0.8)
            except Exception as e:
                print(f"   => Fit error KWW: {e}")
        
        # Plot raw correlation data
        valid_a = (t_fit >= 0)
        t_valid_a = t_fit[valid_a]
        f_s_valid = f_s[valid_a]
        
        lbl_data = r"$\varepsilon=" + f"{eps}$"
        
        marker_style = {0.1: 'o', 0.7: '^', 1.4: 'x'}[eps]
        ms = {0.1: 3, 0.7: 3, 1.4: 3.5}[eps]
        
        # Subplot (a) - Raw relaxation F_s(q, t) vs t (every 2nd point for clarity)
        ax_a.plot(t_valid_a[::2], f_s_valid[::2], marker_style, markersize=ms, color=colors_map[eps], label=lbl_data, alpha=0.6)
        
        # Subplot (b) - Double logarithmic linearization
        valid_b = (t_fit > 0) & (f_s > 0.01)
        t_valid_b = t_fit[valid_b]
        y_valid_b = -np.log(f_s[valid_b] / A_kww)
        ax_b.plot(t_valid_b, y_valid_b, marker_style, markersize=ms, color=colors_map[eps], label=lbl_data, alpha=0.6)
        
    # Format Subplot (a)
    ax_a.set_xlabel(r"Temps $t$ (u.a.)")
    ax_a.set_ylabel(r"Scattering intermédiaire $F_s(q, t)$")
    ax_a.set_xlim(0, max_tau_time)
    ax_a.set_ylim(-0.05, 1.05)
    ax_a.grid(True, which="both", alpha=0.2, ls="--")
    ax_a.legend(loc="upper right", framealpha=0.9)
    ax_a.set_title("a) Décroissance temporelle de $F_s(q, t)$")
    
    # Format Subplot (b)
    ax_b.set_xlabel(r"Temps de corrélation $t$ (u.a.)")
    ax_b.set_ylabel(r"$-\ln\left(F_s(q, t)/A\right)$")
    ax_b.set_xscale('log')
    ax_b.set_yscale('log')
    ax_b.set_xlim(0.1, max_tau_time * 1.1)
    ax_b.set_ylim(0.01, 10.0)
    ax_b.grid(True, which="both", alpha=0.2, ls="--")
    ax_b.legend(loc="lower right", framealpha=0.9)
    ax_b.set_title("b) Linéarisation log-log de la loi KWW")
    
    plt.tight_layout()
    
    pdf_path_report = os.path.join(fig_dir, "17_intermediate_scattering.pdf")
    plt.savefig(pdf_path_report, dpi=300, bbox_inches='tight')
    print(f"\n--> Successfully saved: {pdf_path_report}")

if __name__ == "__main__":
    main()
