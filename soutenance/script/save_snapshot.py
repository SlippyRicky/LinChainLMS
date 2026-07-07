# -*- coding: utf-8 -*-
"""
save_snapshot.py
Saves a static snapshot of the wave propagation animation comparing
ordered vs disordered chains at t = T_snapshot.
The chain starts at rest at t=0 and is excited by a temporal pulse at the center.
The x-axis is relative to the center node.
The chain is placed at the bottom (y=0) and the envelope grows upwards.
Simulates a longer chain (N=160) but only plots 1 in 4 masses for clarity.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Configure premium LaTeX styles
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size": 11,
    "legend.fontsize": 9,
    "axes.titlesize": 11
})

def get_spring_coords(x1, y1, x2, y2, n_turns=8, width=0.06):
    t = np.linspace(0, 1, 2 + 2 * n_turns)
    x = x1 + (x2 - x1) * t
    y = np.zeros_like(x)
    y[0] = y1
    y[-1] = y2
    y[1:-1] = y1 + width * (-1)**np.arange(2 * n_turns)
    return x, y

def main():
    N = 160
    T_snapshot = 0.0
    dt = 0.05
    steps = int(T_snapshot / dt)
    
    center = N // 2
    d = 1.0
    # X-axis relative to the excitation node (center is at 0)
    eq_x = np.arange(N) * d - center * d
    
    # Indices of masses to plot (1 in 4)
    plot_idx = np.arange(0, N, 4)
    
    m_ordered = np.ones(N)
    k_ordered = np.ones(N + 1)
    
    np.random.seed(42)
    epsilon = 1.4
    m_disordered = 1.0 * (1.0 + epsilon * (np.random.rand(N) - 0.2))
    m_disordered = np.clip(m_disordered, 0.2, None)
    k_disordered = np.ones(N + 1)
    
    # Pulse parameters matching physical simulations
    t0 = 20.0
    sigma_t = 4.0
    sigma = 1.0 / (4.0 * sigma_t**2)
    omega_ext = 1.4
    amp = 0.15
    
    def simulate(m, k):
        u = np.zeros(N)
        v = np.zeros(N)
        u_max = np.zeros(N)
        l_abc = 8
        gamma_max = 0.1
        g_profile = np.zeros(N)
        for i in range(l_abc):
            g_profile[i] = gamma_max * ((l_abc - i) / l_abc)**2
            g_profile[N-1-i] = gamma_max * ((l_abc - i) / l_abc)**2
            
        def get_acc(u_c, v_c):
            f = np.zeros(N)
            f[0] = -k[0]*u_c[0] - k[1]*(u_c[0]-u_c[1])
            f[1:N-1] = -k[1:N-1]*(u_c[1:N-1]-u_c[0:N-2]) - k[2:N]*(u_c[1:N-1]-u_c[2:N])
            f[N-1] = -k[N-1]*(u_c[N-1]-u_c[N-2]) - k[N]*u_c[N-1]
            f -= g_profile * v_c
            return f / m
            
        a = get_acc(u, v)
        for step in range(steps):
            t_curr = step * dt
            v_half = v + 0.5 * dt * a
            u += v_half * dt
            
            # Apply imposed displacement at central node
            arg_gauss = -2.0 * sigma * (t_curr + dt - t0)**2
            if arg_gauss > -50.0:
                u[center] = amp * np.exp(arg_gauss) * np.sin(omega_ext * (t_curr + dt - t0))
            else:
                u[center] = 0.0
            v_half[center] = 0.0
            
            # Track running maximum displacement for each mass
            u_max = np.maximum(u_max, np.abs(u))
            
            a = get_acc(u, v_half)
            a[center] = 0.0
            
            v = v_half + 0.5 * dt * a
            v[center] = 0.0
            
        return u, u_max

    u_ord, u_max_ord = simulate(m_ordered, k_ordered)
    u_dis, u_max_dis = simulate(m_disordered, k_disordered)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.0))
    plt.subplots_adjust(hspace=0.6, right=0.78, bottom=0.18, top=0.9)
    
    # Set transparent background
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    
    scale = 3.0  # Visual scaling factor for displacement
    
    # Calculate y-limits based on maximum excitation mass displacement + 10%
    max_disp_excitation_scaled = amp * scale
    ymax = max_disp_excitation_scaled * 1.1  # 10% headroom
    ymin = -0.1 * ymax  # 10% below zero for chain placement at bottom
    
    for ax in (ax1, ax2):
        ax.set_facecolor('none')
        ax.patch.set_alpha(0.0)
        ax.set_xlim(-center - 2, N - center + 2)
        ax.set_ylim(ymin, ymax + 0.5)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.tick_params(axis='x', colors='#666666', labelsize=9)
        ax.set_xlabel(r"Position relative au site d'excitation $j - j_c$", fontsize=9, color='#666666')
        
    ax1.set_title(r"\textbf{Milieu P\'{e}riodique Ordonn\'{e} (Propagation Balistique)}", color='#2C3E50')
    ax2.set_title(r"\textbf{Milieu D\'{e}sordonn\'{e} (D\'{e}sordre de masses ($\varepsilon = %1.1f$))}" % epsilon, color='#2C3E50')
    
    ax1.axhline(0, color='#EEEEEE', linestyle='--', zorder=0)
    ax2.axhline(0, color='#EEEEEE', linestyle='--', zorder=0)
    
    # Ordered
    curr_x_ord = eq_x + u_ord * scale
    for i in range(len(plot_idx) - 1):
        idx1 = plot_idx[i]
        idx2 = plot_idx[i+1]
        sx, sy = get_spring_coords(curr_x_ord[idx1], 0, curr_x_ord[idx2], 0)
        ax1.plot(sx, sy, color='#BBBBBB', linewidth=1, zorder=1)
    
    p1 = ax1.scatter(curr_x_ord[plot_idx], np.zeros(len(plot_idx)), s=40, color='#1F77B4', edgecolor='#115588', zorder=2, label=r"Atome ($M_j = 1.0$)")
    p2, = ax1.plot([], [], color='#BBBBBB', linewidth=1.5, label=r"Liaison \'{e}lastique")
    p3, = ax1.plot([], [], color='#1F77B4', alpha=0.5, linewidth=2, label=r"Enveloppe $U_j^{\max}$")
    
    # Plot smooth envelope over all nodes
    ax1.plot(eq_x, u_max_ord * scale, color='#1F77B4', alpha=0.3, linewidth=2, zorder=0)
    ax1.fill_between(eq_x, 0, u_max_ord * scale, color='#1F77B4', alpha=0.15, zorder=0)
    ax1.legend(handles=[p1, p2, p3], loc='lower left', bbox_to_anchor=(1.02, 0.0), frameon=True, facecolor='white', framealpha=0.9, edgecolor='#E0E0E0')
    
    # Disordered
    curr_x_dis = eq_x + u_dis * scale
    for i in range(len(plot_idx) - 1):
        idx1 = plot_idx[i]
        idx2 = plot_idx[i+1]
        sx, sy = get_spring_coords(curr_x_dis[idx1], 0, curr_x_dis[idx2], 0)
        ax2.plot(sx, sy, color='#BBBBBB', linewidth=1, zorder=1)
    
    sizes_dis = 15 + m_disordered[plot_idx] * 40
    norm = plt.Normalize(np.min(m_disordered), np.max(m_disordered))
    colors_dis = plt.cm.coolwarm(norm(m_disordered[plot_idx]))
    
    scatter_dis = ax2.scatter(curr_x_dis[plot_idx], np.zeros(len(plot_idx)), s=sizes_dis, color=colors_dis, edgecolor='#555555', zorder=2)
    
    p1_dis = ax2.scatter([], [], s=45, color='#F39C12', edgecolor='#555555', label=r"Atome ($M_j$ variable)")
    p2_dis, = ax2.plot([], [], color='#BBBBBB', linewidth=1.5, label=r"Liaison")
    p3_dis, = ax2.plot([], [], color='#D62728', alpha=0.5, linewidth=2, label=r"Enveloppe $U_j^{\max}$")
    
    # Plot smooth envelope over all nodes
    ax2.plot(eq_x, u_max_dis * scale, color='#D62728', alpha=0.3, linewidth=2, zorder=0)
    ax2.fill_between(eq_x, 0, u_max_dis * scale, color='#D62728', alpha=0.15, zorder=0)
    ax2.legend(handles=[p1_dis, p2_dis, p3_dis], loc='lower left', bbox_to_anchor=(1.02, 0.0), frameon=True, facecolor='white', framealpha=0.9, edgecolor='#E0E0E0')
    
    # Colorbar to explain colors of the masses
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.22, 0.04, 0.35, 0.025])
    cbar_ax.set_facecolor('none')
    cbar_ax.patch.set_alpha(0.0)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r"Masse de l'atome $M_j$ (en unit\'{e}s $M_0$)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    script_dir_abs = os.path.dirname(os.path.abspath(__file__))
    soutenance_dir = os.path.abspath(os.path.join(script_dir_abs, ".."))
    
    snapshot_path = os.path.join(soutenance_dir, "wave_propagation_snapshot.png")
    plt.savefig(snapshot_path, dpi=300, transparent=True)
    print(f"Snapshot saved successfully to: {snapshot_path}")

if __name__ == '__main__':
    main()
