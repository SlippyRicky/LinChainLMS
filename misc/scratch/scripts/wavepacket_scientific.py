import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from numba import njit

# 2. PARAMÈTRES PHYSIQUES ---
m0, k0 = 4.66e-26, 50.0
omega_max = 2 * np.sqrt(k0 / m0)
omega_ext = 0.40 * omega_max  
ratio_om = omega_ext / omega_max

N = 4000
DT = 1.0e-15
STEPS = 80000
DOWNSAMPLE = 500
T_PULSE = 3.0e-12
L_ABC = 300
gamma_max = 1.2e-13

@njit
def run_sim_rigorous(is_disordered, node):
    u, v = np.zeros(N), np.zeros(N)
    np.random.seed(42)
    eps = 0.5 if is_disordered else 0.0
    k_bonds = np.ones(N+1) * k0
    if is_disordered:
        for i in range(N+1):
            k_bonds[i] = k0 * (1.0 + eps * (np.random.random() - 0.5))
            
    g_profile = np.zeros(N)
    for i in range(L_ABC):
        g_profile[i] = gamma_max * ((L_ABC - i) / L_ABC)**2
        g_profile[N-1-i] = gamma_max * ((L_ABC - i) / L_ABC)**2
        
    n_frames = STEPS // DOWNSAMPLE
    u_hist = np.zeros((n_frames, N))
    e_hist = np.zeros(n_frames)
    envelope = np.zeros(N) # Calcul de l'enveloppe en temps réel
    
    # Init acc
    def get_a(u_c, v_c, t_c):
        f = np.zeros(N)
        # Forces internes
        for i in range(1, N-1):
            f[i] = -k_bonds[i]*(u_c[i]-u_c[i-1]) - k_bonds[i+1]*(u_c[i]-u_c[i+1])
        f[0] = -k_bonds[0]*u_c[0] - k_bonds[1]*(u_c[0]-u_c[1])
        f[N-1] = -k_bonds[N-1]*(u_c[N-1]-u_c[N-2]) - k_bonds[N]*u_c[N-1]
        # ABC
        for i in range(N): f[i] -= g_profile[i] * v_c[i]
        # Excitation
        if t_c < 2 * T_PULSE:
            amp = np.exp(-(t_c - T_PULSE)**2 / (2 * (T_PULSE/6)**2))
            f[node] += 1.5e-10 * amp * np.sin(omega_ext * t_c)
        return f / m0

    a = get_a(u, v, 0.0)
    for i in range(STEPS):
        t = i * DT
        v_half = v + 0.5 * DT * a
        u += v_half * DT
        
        # Update envelope à CHAQUE pas pour une ligne lisse
        for j in range(N):
            abs_u = abs(u[j])
            if abs_u > envelope[j]: envelope[j] = abs_u
            
        a = get_a(u, v_half, t + DT)
        v = v_half + 0.5 * DT * a
        
        if i % DOWNSAMPLE == 0:
            idx = i // DOWNSAMPLE
            if idx < n_frames:
                u_hist[idx] = u
                # Énergie TOTALE (Cinétique + Potentielle)
                ek = 0.0
                for j in range(N): ek += v[j]**2
                ek *= 0.5 * m0
                
                ep = 0.5 * k_bonds[0] * u[0]**2
                for j in range(1, N):
                    ep += 0.5 * k_bonds[j] * (u[j]-u[j-1])**2
                ep += 0.5 * k_bonds[N] * u[N-1]**2
                
                e_hist[idx] = ek + ep
                
    return u_hist, e_hist, envelope

# --- 3. EXÉCUTION ---
print(f"[SIM] Calcul rigoureux N={N}...")
hA, eA, envA = run_sim_rigorous(False, L_ABC + 100)
hB, eB, envB = run_sim_rigorous(True, L_ABC + 100)
hC, eC, envC = run_sim_rigorous(False, N // 2)
hD, eD, envD = run_sim_rigorous(True, N // 2)

# --- 4. PLOTTING ---
plt.style.use('default')
fig = plt.figure(figsize=(15, 11), facecolor='white')
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.45)
fig.suptitle(fr"$\mathbf{{Dynamique\ du\ paquet\ d'onde\ (Resolution\ de\ Verlet\ rigoureuse)}}$" + "\n" +
             fr"$\mathbf{{Excitation\ :\ \omega_{{ext}} = {ratio_om:.2f}\ \omega_{{max}},\ \tau_{{pulse}} = {T_PULSE*1e12:.1f}\ ps}}$", y=0.98)

axes = [fig.add_subplot(gs[i, j], facecolor='white') for i in range(2) for j in range(2)]
ax_e = fig.add_subplot(gs[2, :], facecolor='white')
titles = [r"$\mathbf{A.\ Ordonne\ :\ Bord}$", r"$\mathbf{B.\ Desordonne\ :\ Bord}$",
          r"$\mathbf{C.\ Ordonne\ :\ Centre}$", r"$\mathbf{D.\ Desordonne\ :\ Centre}$"]
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
lines = []
data_list = [hA, hB, hC, hD]
env_list = [envA, envB, envC, envD]

for i, ax in enumerate(axes):
    ax.set_title(titles[i], pad=10)
    m_val = np.max(env_list[i])
    ax.set_ylim(-m_val * 1.1, m_val * 1.1)
    ax.set_xlim(0, N)
    ax.set_xlabel(r"Position $n$ (Sites)"); ax.set_ylabel(r"Deplacement $u_n$ (m)")
    
    # Tracé de l'enveloppe (LIGNE LISSE)
    ax.plot(np.arange(N), env_list[i], color=colors[i], ls='--', lw=1.2, alpha=0.6, label="Enveloppe")
    ax.plot(np.arange(N), -env_list[i], color=colors[i], ls='--', lw=1.2, alpha=0.6)
    
    l, = ax.plot(np.arange(N), data_list[i][0], color=colors[i], lw=1.0)
    lines.append(l)

ax_e.set_title(r"$\mathbf{Energie\ Totale\ (Cin\acute{{e}}tique\ +\ Potentielle)\ E_{tot}(t)\ (J)}$", pad=10)
e_lines = [ax_e.plot([], [], color=colors[i], label=titles[i], lw=2)[0] for i in range(4)]
ax_e.legend(loc='upper right', fontsize=9); ax_e.set_xlabel(r"Temps $t$ (ps)"); ax_e.set_ylabel(r"Energie (J)")
ax_e.set_xlim(0, STEPS*DT*1e12); ax_e.set_ylim(0, max(np.max(eC), np.max(eD)) * 1.1)

# --- 5. ANIMATION ---
time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, fontweight='bold')
ps_times = np.arange(len(hA)) * DOWNSAMPLE * DT * 1e12
total_frames = len(hA) + 60

def update(frame):
    idx = min(frame, len(hA) - 1)
    for i in range(4):
        lines[i].set_ydata(data_list[i][idx])
        e_lines[i].set_data(ps_times[:idx+1], [eA, eB, eC, eD][i][:idx+1])
    time_text.set_text(fr"$\mathbf{{Temps\ :\ {ps_times[idx]:.2f}\ ps}}$")
    return lines + e_lines + [time_text]

print(f"[ANIM] Generation GIF (Physics-Rigorous)...")
ani = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=True)
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wavepacket_physics_rigorous.gif")
ani.save(save_path, writer='pillow', fps=20, dpi=100, savefig_kwargs={'facecolor': 'white'})
plt.close()
print(f"[DONE] Fichier : {save_path}")
