# -*- coding: utf-8 -*-
"""
generate_animation.py
Generates a side-by-side or top-and-bottom looped animation comparing
wavepacket propagation in an ordered chain vs a disordered chain (Anderson localization).
Uses LaTeX fonts, places legends outside, and saves as:
1. wave_propagation.gif (transparent background)
2. wave_propagation.mp4 (white background for Beamer pdfpc integration)
The envelope shows the cumulative maximum displacement U_j^{max} labeled in legends.
The y-limits are automatically adjusted to 110% of the peak excitation amplitude + 0.5.
Simulates a longer chain (N=160) but only plots 1 in 4 masses for clarity.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    # Parameters
    N = 160
    dt = 0.05
    downsample = 5  # Downsample for animation frame rate
    
    center = N // 2
    d = 1.0
    # X-axis relative to the excitation node (center is at 0)
    eq_x = np.arange(N) * d - center * d
    
    # Indices of masses to plot (1 in 4)
    plot_idx = np.arange(0, N, 4)
    
    # Initialize chains
    m_ordered = np.ones(N)
    k_ordered = np.ones(N + 1)
    
    np.random.seed(42)
    epsilon = 1.0
    m_disordered = 1.0 * (1.0 + epsilon * (np.random.rand(N) - 0.2))
    m_disordered = np.clip(m_disordered, 0.2, None)  # Ensure positive masses
    k_disordered = np.ones(N + 1)
    
    # Pulse parameters matching physical simulations
    t0 = 20.0
    sigma_t = 4.0
    sigma = 1.0 / (4.0 * sigma_t**2)
    omega_ext = 1.1
    amp = 0.15
    
    # Calculate group velocity and travel time to boundaries programmatically
    vg = np.sqrt(1.0 - (omega_ext / 2.0)**2)
    T_max = float(np.ceil(t0 + (N / 2.0) / vg))  # Ensure full travel time to ends
    steps = int(T_max / dt)
    n_frames = steps // downsample
    
    # Simulators
    def simulate(m, k):
        u = np.zeros(N)
        v = np.zeros(N)
        
        # Dampened boundary to avoid reflections (ABC)
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
        history = []
        
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
            
            a = get_acc(u, v_half)
            a[center] = 0.0
            
            v = v_half + 0.5 * dt * a
            v[center] = 0.0
            
            if step % downsample == 0:
                history.append(u.copy())
                
        return np.array(history)

    print("Simulating ordered chain...")
    history_ord = simulate(m_ordered, k_ordered)
    print("Simulating disordered chain...")
    history_dis = simulate(m_disordered, k_disordered)
    
    # Calculate cumulative maximum envelopes dynamically
    cum_max_ord = np.maximum.accumulate(np.abs(history_ord), axis=0)
    cum_max_dis = np.maximum.accumulate(np.abs(history_dis), axis=0)
    
    # Plotting setup
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
    ax2.set_title(r"\textbf{Milieu D\'{e}sordonn\'{e} (D\'{e}sordre de masses (\varepsilon = %1.4f))}", color='#2C3E50')
    
    # Draw equilibrium line
    ax1.axhline(0, color='#EEEEEE', linestyle='--', zorder=0)
    ax2.axhline(0, color='#EEEEEE', linestyle='--', zorder=0)
    
    # Initialize visual elements
    # Case 1: Ordered
    lines_ord = []
    for i in range(len(plot_idx) - 1):
        line, = ax1.plot([], [], color='#BBBBBB', linewidth=1, zorder=1)
        lines_ord.append(line)
    
    scatter_ord = ax1.scatter(eq_x[plot_idx], np.zeros(len(plot_idx)), s=40, color='#1F77B4', edgecolor='#115588', zorder=2)
    envelope_ord, = ax1.plot([], [], color='#1F77B4', alpha=0.3, linewidth=2, zorder=0)
    fill_ord = ax1.fill_between([], [], color='#1F77B4', alpha=0.1, zorder=0)
    
    # Legends for ordered (placed outside bottom-right)
    p1 = ax1.scatter([], [], s=40, color='#1F77B4', edgecolor='#115588', label=r"Atome ($M_j = 1.0$)")
    p2, = ax1.plot([], [], color='#BBBBBB', linewidth=1.5, label=r"Liaison \'{e}lastique")
    p3, = ax1.plot([], [], color='#1F77B4', alpha=0.5, linewidth=2, label=r"Enveloppe $U_j^{\max}$")
    ax1.legend(handles=[p1, p2, p3], loc='lower left', bbox_to_anchor=(1.02, 0.0), frameon=True, facecolor='white', framealpha=0.9, edgecolor='#E0E0E0')
    
    # Case 2: Disordered
    lines_dis = []
    for i in range(len(plot_idx) - 1):
        line, = ax2.plot([], [], color='#BBBBBB', linewidth=1, zorder=1)
        lines_dis.append(line)
        
    sizes_dis = 15 + m_disordered[plot_idx] * 40
    norm = plt.Normalize(np.min(m_disordered), np.max(m_disordered))
    colors_dis = plt.cm.coolwarm(norm(m_disordered[plot_idx]))
    scatter_dis = ax2.scatter(eq_x[plot_idx], np.zeros(len(plot_idx)), s=sizes_dis, color=colors_dis, edgecolor='#555555', zorder=2)
    envelope_dis, = ax2.plot([], [], color='#D62728', alpha=0.3, linewidth=2, zorder=0)
    fill_dis = ax2.fill_between([], [], color='#D62728', alpha=0.1, zorder=0)
    
    # Legends for disordered (placed outside bottom-right)
    p1_dis = ax2.scatter([], [], s=45, color='#F39C12', edgecolor='#555555', label=r"Atome ($M_j$ variable)")
    p2_dis, = ax2.plot([], [], color='#BBBBBB', linewidth=1.5, label=r"Liaison")
    p3_dis, = ax2.plot([], [], color='#D62728', alpha=0.5, linewidth=2, label=r"Enveloppe $U_j^{\max}$")
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
    
    # For updating filled areas, we need to save the collection references
    fill_coll = [None, None]
    
    def update(frame):
        # Case 1: Ordered
        u_ord = history_ord[frame]
        curr_x_ord = eq_x + u_ord * scale
        
        # Update springs
        for i in range(len(plot_idx) - 1):
            idx1 = plot_idx[i]
            idx2 = plot_idx[i+1]
            sx, sy = get_spring_coords(curr_x_ord[idx1], 0, curr_x_ord[idx2], 0)
            lines_ord[i].set_data(sx, sy)
            dist = curr_x_ord[idx2] - curr_x_ord[idx1]
            if dist > 4.2:
                lines_ord[i].set_color('#E74C3C') # Red stretch
            elif dist < 3.8:
                lines_ord[i].set_color('#2980B9') # Blue compress
            else:
                lines_ord[i].set_color('#BBBBBB')
                
        # Update masses
        scatter_ord.set_offsets(np.column_stack((curr_x_ord[plot_idx], np.zeros(len(plot_idx)))))
        
        # Update running maximum envelope (smooth over all nodes)
        env_ord = cum_max_ord[frame] * scale
        envelope_ord.set_data(eq_x, env_ord)
        
        # Case 2: Disordered
        u_dis = history_dis[frame]
        curr_x_dis = eq_x + u_dis * scale
        
        # Update springs
        for i in range(len(plot_idx) - 1):
            idx1 = plot_idx[i]
            idx2 = plot_idx[i+1]
            sx, sy = get_spring_coords(curr_x_dis[idx1], 0, curr_x_dis[idx2], 0)
            lines_dis[i].set_data(sx, sy)
            dist = curr_x_dis[idx2] - curr_x_dis[idx1]
            if dist > 4.2:
                lines_dis[i].set_color('#E74C3C')
            elif dist < 3.8:
                lines_dis[i].set_color('#2980B9')
            else:
                lines_dis[i].set_color('#BBBBBB')
                
        # Update masses
        scatter_dis.set_offsets(np.column_stack((curr_x_dis[plot_idx], np.zeros(len(plot_idx)))))
        
        # Update running maximum envelope (smooth over all nodes)
        env_dis = cum_max_dis[frame] * scale
        envelope_dis.set_data(eq_x, env_dis)
        
        # Redraw filled envelopes
        nonlocal fill_coll
        if fill_coll[0] is not None:
            fill_coll[0].remove()
        if fill_coll[1] is not None:
            fill_coll[1].remove()
            
        fill_coll[0] = ax1.fill_between(eq_x, 0, env_ord, color='#1F77B4', alpha=0.15, zorder=0)
        fill_coll[1] = ax2.fill_between(eq_x, 0, env_dis, color='#D62728', alpha=0.15, zorder=0)
        
        return lines_ord + [scatter_ord, envelope_ord] + lines_dis + [scatter_dis, envelope_dis]
        
    print("Generating animation...")
    ani = animation.FuncAnimation(fig, update, frames=len(history_ord), blit=False, interval=50)
    
    script_dir_abs = os.path.dirname(os.path.abspath(__file__))
    soutenance_dir = os.path.abspath(os.path.join(script_dir_abs, ".."))
    
    # 1. Save as GIF with transparent background
    gif_path = os.path.join(soutenance_dir, "wave_propagation.gif")
    print(f"Saving GIF to: {gif_path}")
    ani.save(gif_path, writer='pillow', fps=20, savefig_kwargs={'transparent': True})
    print("GIF saved successfully!")
    
    # 2. Save as MP4 with solid white background for pdfpc integration
    mp4_path = os.path.join(soutenance_dir, "wave_propagation.mp4")
    print(f"Saving MP4 to: {mp4_path}")
    ani.save(mp4_path, writer='ffmpeg', fps=20, codec='libx264', savefig_kwargs={'facecolor': 'white'})
    print("MP4 saved successfully!")

if __name__ == '__main__':
    main()
