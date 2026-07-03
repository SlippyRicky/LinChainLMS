import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size": 18,
})

def draw_spring(ax, x1, x2, y, num_coils=12, amplitude=0.15, color='#4A90E2', lw=1.5):
    pad = (x2 - x1) * 0.1
    xs = np.linspace(x1 + pad, x2 - pad, num_coils * 2 + 1)
    ys = np.zeros_like(xs) + y
    if len(xs) > 2:
        ys[1::2] += amplitude
        ys[2::2] -= amplitude
    ys[0], ys[-1] = y, y
    xs = np.concatenate(([x1], xs, [x2]))
    ys = np.concatenate(([y], ys, [y]))
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle='round', zorder=2)

def run():
    fig, ax = plt.subplots(figsize=(11.76, 5.42))
    # 1. Geometry Constants
    a = 2.5
    n_masses = 11
    L_ABC = 2  # Absorbing boundary layers size
    center = n_masses // 2
    x_wall = -1.23 # 0,1,2 and 8,9,10
    
    eq_x = np.arange(n_masses) * a
    
    rng = np.random.default_rng(42)
    radii = 0.25 + 0.05 * (rng.random(n_masses) - 0.5)
    spring_lws = 1.5 + 1.5 * (rng.random(n_masses) - 0.5)
    spring_amps = 0.15 + 0.05 * (rng.random(n_masses) - 0.5)
    
    # Add structural disorder (springs) and mass disorder
    np.random.seed(42)
    spring_lws[L_ABC:n_masses-L_ABC-1] += np.random.uniform(-1.0, 1.5, n_masses - 2*L_ABC - 1)
    spring_amps[L_ABC:n_masses-L_ABC-1] += np.random.uniform(-0.08, 0.08, n_masses - 2*L_ABC - 1)
    radii[L_ABC:n_masses-L_ABC] += np.random.uniform(-0.1, 0.1, n_masses - 2*L_ABC)  
    
    # Draw Springs
    for i in range(n_masses - 1):
        start_x = eq_x[i] + radii[i]
        end_x = eq_x[i+1] - radii[i+1]
        c = '#555555' if (i < L_ABC or i >= n_masses - L_ABC - 1) else '#4A90E2'
        lw = 1.5 if (i < L_ABC or i >= n_masses - L_ABC - 1) else spring_lws[i]
        draw_spring(ax, start_x, end_x, 0, lw=lw, amplitude=spring_amps[i], color=c)

    # Draw Masses
    for i in range(n_masses):
        color = '#27ae60' if (i < L_ABC or i >= n_masses - L_ABC) else '#4682B4'
        if i == center: color = '#D32F2F' # Source
        circle = patches.Circle((eq_x[i], 0), radius=radii[i], facecolor=color, edgecolor='black', lw=2, zorder=10)
        ax.add_patch(circle)
        
    # 6. ABC Zones (Green overlays)
    abc_left_start = x_wall
    abc_left_end = eq_x[L_ABC-1] + 1.0
    rect_l = patches.Rectangle((abc_left_start, -2.0), abc_left_end - abc_left_start, 4.2, 
                             facecolor='#27ae60', alpha=0.15, edgecolor='none', zorder=1)
    ax.add_patch(rect_l)
    ax.text((abc_left_start + abc_left_end)/2, -1.2, "Couche\nAbsorbante", color='#1e8449', ha='center', va='center', fontweight='bold', fontsize=9)

    abc_right_start = eq_x[n_masses - L_ABC] - 1.0
    abc_right_end = eq_x[-1] + 0.8
    rect_r = patches.Rectangle((abc_right_start, -2.0), abc_right_end - abc_right_start, 4.2, 
                             facecolor='#27ae60', alpha=0.15, edgecolor='none', zorder=1)
    ax.add_patch(rect_r)
    ax.text((abc_right_start + abc_right_end)/2, -1.2, "Couche\nAbsorbante", color='#1e8449', ha='center', va='center', fontweight='bold', fontsize=9)

    # Draw Central Excitation
    ax.annotate('', xy=(eq_x[center], 0.4), xytext=(eq_x[center], 0.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='#D32F2F'))
    ax.text(eq_x[center], 0.85, "Noeud d'Excitation\n$n_0 = N/2$", ha='center', va='bottom', color='#D32F2F', fontweight='bold')
    
    # Draw Wavepacket/Harmonic icon above it
    x_wave = np.linspace(eq_x[center] - 2.5, eq_x[center] + 2.5, 150)
    # Physically correct form: Gaussian envelope * cosine carrier, 20% smaller amplitude
    y_wave = 2.4 + 0.32 * np.exp(-(x_wave - eq_x[center])**2 / (2.0 * 0.8**2)) * np.cos(3.5 * (x_wave - eq_x[center]))
    ax.plot(x_wave, y_wave, color='#D32F2F', lw=2)
    ax.text(eq_x[center], 2.8, r"Excitation temporelle $F_{\text{ext}}(t)$", ha='center', va='bottom', color='#D32F2F', fontsize=10)

    # Draw Disordered Bulk label
    ax.text((eq_x[L_ABC] + eq_x[n_masses - L_ABC - 1])/2, -0.6, "Région Centrale Désordonnée\n" + r"$K_n = K_0(1 + \varepsilon X_n)$" + "\n" + r"$m_n = m_0(1 + \varepsilon Y_n)$", ha='center', va='top', fontsize=10)

    # Lattice Constant
    ax.annotate('', xy=(eq_x[center-2], 0.6), xytext=(eq_x[center-1], 0.6), arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text((eq_x[center-2] + eq_x[center-1])/2, 0.7, r'$a$', ha='center', va='bottom', fontsize=10)

    # Styling
    ax.set_xlim(abc_left_start - 0.2, abc_right_end + 0.2)
    ax.set_ylim(-2.2, 3.8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    # Save to output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import plot_config
    fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
    # Directory creation handled by plot_config
    output_path = os.path.join(fig_dir, "04_disordered_chain_schematic.pdf")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"--> Saved figure to {output_path}")

if __name__ == '__main__':
    run()
