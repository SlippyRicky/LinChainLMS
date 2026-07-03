import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- STYLE CONFIGURATION (Premium Manuscript) ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 22,             # Tick labels (scales to ~5.9pt on paper)
    "axes.labelsize": 26,        # Axes labels (scales to ~7.0pt on paper)
    "axes.titlesize": 32,        # Title size (scales to ~8.6pt on paper)
    "legend.fontsize": 22,       # Legend size (scales to ~5.9pt on paper)
    "lines.linewidth": 2.2,      # Thicker lines for print
    "axes.grid": False,
})

C_BAL = "#2ca02c"  # Green for Ballistic/Perfect Crystal
C_DIF = "#1f77b4"  # Professional Blue
C_LOC = "#d62728"  # Professional Red
C_SLAB = "#f8f9f9" # Very light gray for slab background
C_TRANS = "#2e86c1" # Darker blue for transducer

def draw_regime(ax, title, regime_type='diffusive'):
    # regime_type can be 'ballistic', 'diffusive', 'localized'
    ax.set_xlim(-1.5, 15.5)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect('equal')
    
    # Make them real plots with axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_bounds(0, 8.0) # Constrain bottom axis to the slab
    ax.set_ylabel(r"Pos. transverse $x_\perp$", fontsize=26)
    ax.set_xticks([0, 4.0, 8.0]) # Only draw ticks within the slab
    ax.set_xticklabels([]) # Hide x tick labels (numeric values)
    if regime_type == 'localized':
        ax.text(4.0, -4.0, r"Distance de propagation $z$", fontsize=26, ha='center', va='top')
    ax.set_title(title, fontweight='bold', pad=20)
    
    # 1. Draw Slab Container
    slab_x = 0.0
    slab_w = 8.0
    slab_y = -2.5
    slab_h = 5.0
    
    slab = patches.Rectangle((slab_x, slab_y), slab_w, slab_h, 
                             facecolor=C_SLAB, edgecolor='black', lw=1.5, zorder=1)
    ax.add_patch(slab)
    
    # 2. Add scattered beads/defects
    if regime_type in ['diffusive', 'localized']:
        rng_beads = np.random.default_rng(2026)
        for _ in range(120):
            bx = rng_beads.uniform(slab_x + 0.2, slab_x + slab_w - 0.2)
            by = rng_beads.uniform(slab_y + 0.2, slab_y + slab_h - 0.2)
            ax.plot(bx, by, '.', color='gray', alpha=0.35, markersize=6, zorder=2)
    
    # 3. Draw Transducer
    trans_w = 0.8
    trans_h = 1.0
    trans_x = slab_x - trans_w
    trans_y = - trans_h / 2.0
    
    transducer = patches.Rectangle((trans_x, trans_y), trans_w, trans_h, 
                                   facecolor=C_TRANS, edgecolor='black', lw=1.2, zorder=5)
    ax.add_patch(transducer)
    ax.text(trans_x + trans_w/2.0, 0, "Source", color='white', 
            ha='center', va='center', fontsize=18, fontweight='bold', rotation=90)
    
    # 4. Wave Paths
    rng = np.random.default_rng(42) if regime_type == 'diffusive' else np.random.default_rng(101)
    n_paths = 7
    
    for i in range(n_paths):
        z = np.linspace(0, slab_w, 150)
        
        if regime_type == 'ballistic':
            # Waves fan outwards from the point source (diffraction / geometric spreading)
            spread_max = 1.5
            y = spread_max * (i - (n_paths - 1)/2.0) / (n_paths/2.0) * (z / slab_w)
            color = C_BAL
            alpha = 0.8
        elif regime_type == 'diffusive':
            spread = 1.8 * np.sqrt(z / slab_w)
            noise = np.cumsum(rng.normal(0, 0.1, len(z)))
            noise = noise - noise[0]
            y = spread * (i - (n_paths - 1)/2.0) / (n_paths/2.0) + noise * 0.5
            color = C_DIF
            alpha = 0.7
        elif regime_type == 'localized':
            # Starts at 0 and saturates (transverse confinement)
            target_y = 0.5 * (i - (n_paths - 1)/2.0) / (n_paths/2.0)
            y = target_y * (1 - np.exp(-z/1.0))
            noise = np.sin(z * 4.0 + i) * 0.15 * (1 - np.exp(-z/0.5)) + rng.normal(0, 0.05, len(z)) * (1 - np.exp(-z/0.5))
            y = y + noise
            color = C_LOC
            alpha = 0.8
            
        ax.plot(z, y, color=color, alpha=alpha, lw=2.5, zorder=3)

    # 5. Transverse Profiles
    exit_x = slab_x + slab_w
    profile_start_x = exit_x + 0.5
    y_profile = np.linspace(-2.5, 2.5, 300)
    
    if regime_type == 'ballistic':
            sigma = 0.6
            amp = 2.0
            I_profile = profile_start_x + amp * np.exp(-y_profile**2 / (2 * sigma**2))
            color = C_BAL
            text_label = r"\textbf{Transport Balistique}" + "\n" + r"(Élargissement géométrique linéaire)"
    elif regime_type == 'diffusive':
        sigma = 1.4
        amp = 2.0
        I_profile = profile_start_x + amp * np.exp(-y_profile**2 / (2 * sigma**2))
        color = C_DIF
        text_label = r"\textbf{Diffusion Transverse}" + "\n" + r"(Élargissement $\sigma^2 \propto z$)"
    elif regime_type == 'localized':
        xi = 0.35
        amp = 2.0
        I_profile = profile_start_x + amp * np.exp(-np.abs(y_profile) / xi)
        color = C_LOC
        text_label = r"\textbf{Localisation Transverse}" + "\n" + r"(Confinement $\sigma^2 \to \mathrm{cste}$)"
        
    ax.plot(I_profile, y_profile, color=color, lw=4.0, zorder=4)
    ax.fill_betweenx(y_profile, profile_start_x, I_profile, color=color, alpha=0.2, zorder=2)
    
    ax.plot([profile_start_x, profile_start_x], [-2.5, 2.5], 'k-', lw=2.2, alpha=0.7)
    ax.plot([profile_start_x, profile_start_x + 2.5], [0, 0], 'k:', lw=2.0, alpha=0.5)
    
    ax.text(profile_start_x + 0.8, 1.8, text_label, color=color, fontsize=22, ha='left')

def draw_variance_plot(ax):
    z = np.linspace(0, 160, 200)
    
    # Ballistic
    var_bal = 0.25 * z**2 + 15
    ax.plot(z, var_bal, ':', color='gray', lw=2.5, label=r"Balistique ($\sigma^2 \propto z^2$)")
    
    # Diffusive
    D = 4.78
    var_dif = D * z + 15
    ax.plot(z, var_dif, '-', color=C_DIF, lw=3.8, label=r"Diffusif ($\sigma^2 \propto z$)")
    
    # Localized
    var_loc = 180 - 165 * np.exp(-z/35)
    ax.plot(z, var_loc, '-', color=C_LOC, lw=4.2, label=r"Localisation ($\sigma^2 \to \xi^2$)")
    
    # Annotations
    ax.annotate(r"Saturation ($\sigma \to \xi$)", xy=(130, 178), xytext=(90, 320),
                arrowprops=dict(arrowstyle="->", color=C_LOC, connectionstyle="arc3,rad=0.2", lw=2.2),
                color=C_LOC, fontsize=22)
    
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 850)
    ax.set_xlabel(r"Distance de propagation $z$")
    ax.set_ylabel(r"Variance transverse $\sigma^2$")
    ax.set_title(r"\textbf{D. Évolution de la Variance}", fontweight='bold', pad=20)
    
    ax.legend(loc='upper left', framealpha=0.9, fontsize=22)
    ax.grid(True, alpha=0.2, ls='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def run():
    fig = plt.figure(figsize=(23.52, 13.26))
    
    # 3x2 Grid: A, B, C on the left, D on the right spanning all rows
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.0], wspace=0.08, hspace=0.55)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[:, 1])
    
    draw_regime(ax1, r"\textbf{A. Cristal Parfait}", regime_type='ballistic')
    draw_regime(ax2, r"\textbf{B. Désordre Faible}", regime_type='diffusive')
    draw_regime(ax3, r"\textbf{C. Désordre Fort}", regime_type='localized')
    draw_variance_plot(ax4)
    
    # Manually adjust spacing to avoid overlaps and clipping
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.55, wspace=0.08)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import plot_config
    fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
    output_path = os.path.join(fig_dir, "01_historical_context.pdf")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"--> Figure saved to: {output_path}")

if __name__ == '__main__':
    run()
