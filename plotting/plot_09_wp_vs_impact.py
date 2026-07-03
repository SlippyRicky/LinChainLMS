import matplotlib.pyplot as plt
import numpy as np
import os

# --- STYLE CONFIGURATION ---
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

script_dir = os.path.dirname(os.path.abspath(__file__))
import plot_config
fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
# Directory creation handled by plot_config
data_dir = plot_config.DATA_DIR

def plot_wp_comparison():
    data_path = os.path.join(data_dir, "wp_vs_impact_comparison.npz")
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return
        
    data_load = np.load(data_path)
    times = data_load['times']
    N = int(data_load['N'])
    T_max = int(data_load['T_max'])
    
    fig, axes = plt.subplots(1, 3, figsize=(11.76, 4.66), sharey=True)
    
    scenarios = [
        ("eps_0_1", r"(a)($\varepsilon = 0.1$)"),
        ("eps_0_3", r"(b)($\varepsilon = 0.3$)"),
        ("eps_1_4", r"(c)($\varepsilon = 1.4$)")
    ]
    
    for i, (key, title) in enumerate(scenarios):
        ax = axes[i]
        data = data_load[key]
        
        # Gamma correction for better contrast in tails
        vmax = np.percentile(data, 99.5)
        
        # Shift coordinates to be relative to the center
        center = N // 2
        im = ax.imshow(data, extent=[-center, N - center, T_max, 0], aspect='auto', cmap='magma', 
                       origin='upper', vmin=0, vmax=vmax)
        
        # Set fixed physical zoom window
        ax.set_xlim(-1200, 1200)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(r"Position relative $n - n_0$")
        if i == 0:
            ax.set_ylabel("Temps $t$")

    # Add single colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"Densité d'énergie $\langle |u_n|^2 \rangle$")
    
    save_path = os.path.join(fig_dir, "09_wp_vs_impact.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"--> Generated comparison figure: {save_path}")

if __name__ == '__main__':
    plot_wp_comparison()
