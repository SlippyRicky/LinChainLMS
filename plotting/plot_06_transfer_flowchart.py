import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import os
from scipy.linalg import eigh
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- STYLE CONFIGURATION (Premium Manuscript) ---
plt.rcParams.update({
    "text.usetex": True,
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

# Colors
C_ACOUS = "#1f77b4" # Professional Blue
C_OPTIC = "#d62728" # Professional Red
C_GAP   = "#e0e0e0" # Light Gray

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
import plot_config
fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
# Directory creation handled by plot_config
data_dir = plot_config.DATA_DIR




def plot_transfer_flowchart():
    """Figure 4: Formalisme spatial des matrices de transfert (Scientific Manuscript Style)"""
    fig, ax = plt.subplots(figsize=(11.76, 4.66))
    
    # Coordinates
    y_nodes = 0
    y_inputs = 1.2
    x_states = [-4, 0, 4]
    x_ops = [-2, 2]
    
    # 1. State Vectors (u_n) - Clean Circles
    labels = [r'$\mathbf{u}_{n-1}$', r'$\mathbf{u}_n$', r'$\mathbf{u}_{n+1}$']
    for x, label in zip(x_states, labels):
        # Clean white circle with sharp border
        circle = patches.Circle((x, y_nodes), 0.45, facecolor='white', edgecolor='black', lw=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y_nodes, label, ha='center', va='center', fontsize=17, zorder=6)

    # 2. Transfer Operators (T_n) - Rectangles
    op_labels = [r'$\mathbf{T}_n(K_n)$', r'$\mathbf{T}_{n+1}(K_{n+1})$']
    colors = ['#e8f4f8', '#fdeaea']
    edge_colors = [C_ACOUS, C_OPTIC]
    
    for x, label, face, edge in zip(x_ops, op_labels, colors, edge_colors):
        # Light fill for blocks
        rect = patches.FancyBboxPatch((x-0.7, y_nodes-0.4), 1.4, 0.8, 
                                     boxstyle="round,pad=0.05",
                                     facecolor=face, edgecolor=edge, lw=1.2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y_nodes, label, ha='center', va='center', fontsize=15, color=edge, zorder=4)

    # 3. Connectivity (Arrows)
    arrow_props = dict(arrowstyle='-|>', color='black', lw=1.5, mutation_scale=20)
    
    # u_{n-1} -> T_n
    ax.annotate('', xy=(x_ops[0]-0.7, y_nodes), xytext=(x_states[0]+0.45, y_nodes), arrowprops=arrow_props)
    # T_n -> u_n
    ax.annotate('', xy=(x_states[1]-0.45, y_nodes), xytext=(x_ops[0]+0.7, y_nodes), arrowprops=arrow_props)
    # u_n -> T_{n+1}
    ax.annotate('', xy=(x_ops[1]-0.7, y_nodes), xytext=(x_states[1]+0.45, y_nodes), arrowprops=arrow_props)
    # T_{n+1} -> u_{n+1}
    ax.annotate('', xy=(x_states[2]-0.45, y_nodes), xytext=(x_ops[1]+0.7, y_nodes), arrowprops=arrow_props)

    # 4. Disorder Inputs (Vertical Arrows)
    disorder_labels = [r'$\varepsilon_n$', r'$\varepsilon_{n+1}$']
    for x, label in zip(x_ops, disorder_labels):
        ax.annotate(r'Désordre ' + label, 
                    xy=(x, y_nodes+0.4), xytext=(x, y_inputs),
                    ha='center', va='bottom', fontsize=15, color='#7f8c8d',
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1, ls='--'))

    # 5. Furstenberg Bracket
    bracket_y = -1.0
    ax.plot([x_states[0], x_states[0], x_states[2], x_states[2]], 
            [bracket_y+0.2, bracket_y, bracket_y, bracket_y+0.2], 
            color='black', lw=1.5)
    ax.text((x_states[0]+x_states[2])/2, bracket_y-0.5, 
            r"\textbf{Théorème de Furstenberg} : $\displaystyle \left| \prod_{i=1}^N \mathbf{T}_i \right| \propto \exp(\gamma N)$",
            ha='center', va='center', fontsize=15)

    # Ellipses for continuity
    ax.text(x_states[0]-1.2, y_nodes, r'$\dots$', fontsize=24, ha='center', color='gray')
    ax.text(x_states[2]+1.2, y_nodes, r'$\dots$', fontsize=24, ha='center', color='gray')

    # Final styling
    ax.set_xlim(-6, 6)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "06_transfer_flowchart.pdf"), dpi=300, bbox_inches='tight')
    print("--> Scientific: transfer_flowchart.pdf")


if __name__ == '__main__':
    print('Running plot_transfer_flowchart...')
    plot_transfer_flowchart()
    print('Done.')
