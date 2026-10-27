import subprocess
import sys
import os

## Heatmap deveient longeur de penetration en fonction de la frequence d'excitation et du desordre  (distance a lequelle on atteint Energie max en m=0 / e^1 )


# =================================================================
# SECTION 1 : AUTO-INSTALLATEUR (Pour une installation vierge)
# =================================================================
def bootstrap():
    """Vérifie et installe les bibliothèques nécessaires automatiquement."""
    # Liste des bibliothèques (nom d'import : nom de package pip)
    dependencies = {
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "numba": "numba",
        "PIL": "pillow"
    }
    
    missing_pkgs = []
    for lib, pkg in dependencies.items():
        try:
            __import__(lib)
        except ImportError:
            missing_pkgs.append(pkg)
    
    if missing_pkgs:
        print(f"--- DÉPENDANCES MANQUANTES DÉTECTÉES : {missing_pkgs} ---")
        print("Installation automatique en cours (nécessite une connexion internet)...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_pkgs])
            print("--- INSTALLATION RÉUSSIE ---")
            print("Veuillez relancer le script pour lancer la simulation.")
            sys.exit(0)
        except Exception as e:
            print(f"Erreur lors de l'installation : {e}")
            sys.exit(1)

# Lancement du bootstrap
bootstrap()

# =================================================================
# SECTION 2 : SIMULATION PHYSIQUE (Pure Math Units)
# =================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit

# Paramètres optimisés
plt.rcParams.update({
    "text.usetex": True,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 9,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
    "axes.grid": True,
    "grid.alpha": 0.3
})

# Paramètres
EPS_DISORDER = 0.7
m0, k0 = 1.0, 1.0
omega_max = 2.0
omega_ext = 0.25 * omega_max # T_wave ~ 12.5 u.a.

N = 1500
DT = 0.05           
STEPS = 40000       # Temps total = 2000 u.a. (L'onde peut parcourir les 1500 sites)
DOWNSAMPLE = 200    # ~200 images pour le GIF
T_PULSE = 80.0      
L_ABC = 100
gamma_max = 0.1

@njit
def run_sim(is_disordered, node):
    u, v = np.zeros(N), np.zeros(N)
    np.random.seed(42)
    eps = EPS_DISORDER if is_disordered else 0.0
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
    envelope = np.zeros(N)
    
    def get_a(u_c, v_c, t_c):
        f = np.zeros(N)
        for i in range(1, N-1):
            f[i] = -k_bonds[i]*(u_c[i]-u_c[i-1]) - k_bonds[i+1]*(u_c[i]-u_c[i+1])
        f[0] = -k_bonds[0]*u_c[0] - k_bonds[1]*(u_c[0]-u_c[1])
        f[N-1] = -k_bonds[N-1]*(u_c[N-1]-u_c[N-2]) - k_bonds[N]*u_c[N-1]
        for i in range(N): f[i] -= g_profile[i] * v_c[i]
        if t_c < 2 * T_PULSE:
            amp = np.exp(-(t_c - T_PULSE)**2 / (2 * (T_PULSE/6)**2))
            f[node] += 1.0 * amp * np.sin(omega_ext * t_c)
        return f / m0

    a = get_a(u, v, 0.0)
    for i in range(STEPS):
        t = i * DT
        v_half = v + 0.5 * DT * a
        u += v_half * DT
        for j in range(N):
            abs_u = abs(u[j])
            if abs_u > envelope[j]: envelope[j] = abs_u
        a = get_a(u, v_half, t + DT)
        v = v_half + 0.5 * DT * a
        if i % DOWNSAMPLE == 0:
            idx = i // DOWNSAMPLE
            if idx < n_frames:
                u_hist[idx] = u
                ek = 0.5 * m0 * np.sum(v**2)
                ep = 0.5 * k_bonds[0] * u[0]**2
                for j in range(1, N): ep += 0.5 * k_bonds[j] * (u[j]-u[j-1])**2
                ep += 0.5 * k_bonds[N] * u[N-1]**2
                e_hist[idx] = ek + ep
    return u_hist, e_hist, envelope

# --- EXÉCUTION ---
print(f"Calcul de la dynamique (N={N})...")
hA, eA, envA = run_sim(False, L_ABC + 50)
hB, eB, envB = run_sim(True, L_ABC + 50)
hC, eC, envC = run_sim(False, N // 2)
hD, eD, envD = run_sim(True, N // 2)

# --- VISUALISATION ---
plt.style.use('default')
fig = plt.figure(figsize=(14, 10), facecolor='white')
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.4)
fig.suptitle(r"$\mathbf{Simulation\ du\ Paquet\ d'Onde\ }$" + "\n" +
             fr"$\mathbf{{Param\grave{{e}}tres\ :\ \omega_{{ext}} = {omega_ext/omega_max:.2f}\ \omega_{{max}},\ \epsilon = {EPS_DISORDER}}}$", fontsize=15)

axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
ax_e = fig.add_subplot(gs[2, :])
titles = [r"$\mathbf{A.\ Ordonne\ :\ Bord}$", r"$\mathbf{B.\ Desordonne\ :\ Bord}$",
          r"$\mathbf{C.\ Ordonne\ :\ Centre}$", r"$\mathbf{D.\ Desordonne\ :\ Centre}$"]
colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
lines = []
data_list, env_list = [hA, hB, hC, hD], [envA, envB, envC, envD]

for i, ax in enumerate(axes):
    ax.set_title(titles[i], pad=10)
    m_val = np.max(env_list[i])
    ax.set_ylim(-m_val * 1.1, m_val * 1.1)
    ax.set_xlim(0, N)
    ax.set_xlabel(r"Position $n$ (Sites)")
    ax.set_ylabel(r"Deplacement $u_n$ (u.a.)")
    ax.plot(np.arange(N), env_list[i], color=colors[i], ls='--', lw=1.2, alpha=0.5)
    ax.plot(np.arange(N), -env_list[i], color=colors[i], ls='--', lw=1.2, alpha=0.5)
    l, = ax.plot(np.arange(N), data_list[i][0], color=colors[i], lw=1.0)
    lines.append(l)

ax_e.set_title(r"$\mathbf{Energie\ Totale\ E_{tot}(t)\ (u.a.)}$", pad=10)
e_lines = [ax_e.plot([], [], color=colors[i], label=titles[i], lw=2)[0] for i in range(4)]
ax_e.legend(loc='upper right', fontsize=9)
ax_e.set_xlabel(r"Temps $t$ (u.a.)")
ax_e.set_ylabel(r"Energie")
ax_e.set_xlim(0, STEPS*DT)
ax_e.set_ylim(0, max(np.max(eC), np.max(eD)) * 1.1)

# --- ANIMATION ---
def update(frame):
    idx = min(frame, len(hA) - 1)
    for i in range(4):
        lines[i].set_ydata(data_list[i][idx])
        e_lines[i].set_data(np.arange(idx+1)*DOWNSAMPLE*DT, [eA, eB, eC, eD][i][:idx+1])
    return lines + e_lines

print("Génération du GIF (Patience, cela peut prendre 1 minute)...")
ani = FuncAnimation(fig, update, frames=len(hA)+40, interval=50, blit=True)
filename = "attenuation_comparaison_wavepacket_eps="+str(EPS_DISORDER)+"_N="+str(N)+".gif"
ani.save(filename, writer='pillow', fps=10)
print("--- TERMINÉ : Fichier '" + filename + "' créé ---")
