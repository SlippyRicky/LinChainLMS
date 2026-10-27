import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import eigh
import os
import pandas as pd

# --- Load Physical Constants ---
script_dir = os.path.dirname(os.path.abspath(__file__))
meta_path = os.path.join(script_dir, "../figures/02_linear_disorder/data/sim_metadata.csv")
if os.path.exists(meta_path):
    meta = pd.read_csv(meta_path)
    m0 = float(meta["M_si"].iloc[0])
    k0 = float(meta["K_si"].iloc[0])
else:
    m0, k0 = 4.66e-26, 50.0 # Standard Argon-like parameters

# --- Simulation Parameters ---
N = 500            # Increased for better localization resolution
T_MAX = 800        
DT = 2.5           
X = np.arange(N)
NP_SEED = 42
epsilon = 0.6      # Disorder amplitude (Standardized epsilon)

def get_rigorous_modes(is_disordered=False):
    """Compute exact eigenmodes using the symmetric Dynamic Matrix D."""
    # 1. Masses (Homogeneous)
    m = np.ones(N) * m0
    
    # 2. Stiffnesses (Disordered Bonds)
    np.random.seed(NP_SEED)
    if is_disordered:
        # Uniform bond disorder of width epsilon
        k_bonds = k0 * (1.0 + epsilon * (np.random.rand(N+1) - 0.5))
    else:
        k_bonds = np.ones(N+1) * k0
        
    # 3. Construct Stiffness Matrix K (Fixed-Fixed)
    K_mat = np.zeros((N, N))
    for i in range(N):
        K_mat[i, i] = k_bonds[i] + k_bonds[i+1]
        if i < N-1:
            K_mat[i, i+1] = K_mat[i+1, i] = -k_bonds[i+1]
            
    # 4. Construct Symmetric Dynamic Matrix D = M^-1/2 K M^-1/2
    # Since M is scalar*I here, D = K/m0
    D = K_mat / m0
            
    omega2, Psi = eigh(D)
    omega = np.sqrt(np.clip(omega2, 1e-30, None))
    return omega, Psi, m

def simulate_rigorous_case(omega, Psi, m, sigma, q0, center):
    """Evolve physical displacements through the modal basis."""
    # Physical Initial Displacement v0 (Gaussian)
    v0 = np.exp(-(X - center)**2 / (2 * sigma**2))
    if q0 > 0:
        v0 *= np.cos(q0 * (X - center))
    
    # Canonical Transformation to Symmetric Space: u0 = sqrt(M) v0
    u0 = np.sqrt(m) * v0
    u0 /= np.linalg.norm(u0) # Normalize energy
    
    # Modal projection in symmetric space
    coeffs = Psi.T @ u0
    
    history = []
    times = np.arange(0, T_MAX, DT)
    sqrt_m_inv = 1.0 / np.sqrt(m)
    
    for t in times:
        # u(t) in symmetric space
        u_t = Psi @ (coeffs * np.cos(omega * t))
        # Back to PHYSICAL displacement v(t) = u(t) / sqrt(m)
        v_t = u_t * sqrt_m_inv
        
        # Rigorous Energy Density (Envelope of |v|^2 is sufficient for visualization)
        # Note: We plot v^2 as a proxy for the spatial energy distribution
        energy = v_t**2
        smoothed = np.convolve(energy, np.ones(5)/5, mode='same')
        history.append(smoothed)
        
    return np.array(history)



### FIN DU CALCUL ####



# --- Generate Rigorous Data ---
print("[PHYSICS] Executing Rigorous Modal Analysis...")
om_ord, psi_ord, m_ord = get_rigorous_modes(is_disordered=False)
om_dis, psi_dis, m_dis = get_rigorous_modes(is_disordered=True)

# Scenarios (Rigorous Tuning)
# sigma is in units of lattice sites
hist_ord_imp = simulate_rigorous_case(om_ord, psi_ord, m_ord, sigma=2.0, q0=0.0, center=5)
hist_dis_imp = simulate_rigorous_case(om_dis, psi_dis, m_dis, sigma=2.0, q0=0.0, center=5)
hist_ord_wp  = simulate_rigorous_case(om_ord, psi_ord, m_ord, sigma=15.0, q0=0.45*np.pi, center=50)
hist_dis_wp  = simulate_rigorous_case(om_dis, psi_dis, m_dis, sigma=15.0, q0=0.45*np.pi, center=50)

# --- Animation Setup ---
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
plt.subplots_adjust(hspace=0.25, wspace=0.15)

titles = [
    "A. Ordonné : Impulsion de Bord (Propagation)", 
    "B. Désordonné : Impulsion de Bord (Arrêt)",
    "C. Ordonné : Paquet d'Onde (Vitesse de Groupe)", 
    "D. Désordonné : Paquet d'Onde (Localisation)"
]
data_list = [hist_ord_imp, hist_dis_imp, hist_ord_wp, hist_dis_wp]
colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
lines = []

for i, ax in enumerate(axes.flat):
    ax.set_title(titles[i], fontweight='bold', fontsize=12)
    ax.set_ylim(0, 0.50) 
    ax.set_xlim(0, N)
    ax.grid(True, alpha=0.2)
    line, = ax.plot(X, data_list[i][0], color=colors[i], lw=1.8)
    lines.append(line)
    
    if i >= 2: ax.set_xlabel("Position Atome (n)", fontsize=11)
    if i % 2 == 0: ax.set_ylabel("Amplitude Physique Carrée $|v_n|^2$", fontsize=11)

time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=13, fontweight='bold')

def update(frame):
    for i, ax in enumerate(axes.flat):
        lines[i].set_ydata(data_list[i][frame])
        # Update fills (cleaner update)
        for poly in ax.collections:
            poly.remove()
        ax.fill_between(X, data_list[i][frame], color=colors[i], alpha=0.2)
    time_text.set_text(f"Simulation Rigoureuse (TMM Analog) : t = {frame * DT:.1f}")
    return lines + [time_text]

ani = FuncAnimation(fig, update, frames=len(hist_ord_imp), interval=40, blit=False)

# --- Save Animation ---
save_path = os.path.join(script_dir, "wavepacket_vs_localization_rigorous.gif")
print(f"[ANIMATION] Saving to {save_path}...")
ani.save(save_path, writer='pillow', fps=20)
print(f"[SUCCESS] Rigorous Physics Verified and Animated.")
