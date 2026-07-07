"""
Purpose: Generates the slides-adapted version of the wavepacket envelope profile figure.
Inputs: Read ensemble-averaged wavepacket envelope CSV files (wp_envelopes_eps={eps}_N=*.csv) from `data/`.
Outputs: Produces 08_wavepacket_envelopes_slides.pdf and 08_wavepacket_envelopes_slides.png in `report/figures/slides/`.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob
import re
import json

# Ensure we can import plot_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import plot_config

# --- STYLE CONFIGURATION (Optimized for Presentation Slides) ---
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "legend.fontsize": 12,
    "lines.linewidth": 2.5,  # Thicker lines for readability from afar
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--"
})

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../../"))
slides_fig_dir = os.path.join(plot_config.FIGURES_ROOT, 'slides')

def find_data_file(eps):
    """Dynamically finds the data file matching the disorder level, selecting the one with largest N."""
    data_dir = plot_config.DATA_DIR
    pattern = os.path.join(data_dir, f"wp_envelopes_eps={eps}_N=*.csv")
    files = glob.glob(pattern)
    if files:
        def get_n_val(f):
            m = re.search(r"N=(\d+)", f)
            return int(m.group(1)) if m else 0
        files.sort(key=get_n_val, reverse=True)
        return files[0]
    return None

def plot_slides_envelopes_vertical():
    disorder_levels = [0.1, 0.7, 1.4]
    freq_indices = [1, 5, 10, 15, 19]

    colors = ['#1f77b4', '#33a02c', '#ff7f0e', '#e31a1c', '#6a3d9a']
    
    # 3-row, 1-column layout, wider aspect ratio to prevent overlapping labels and squishing
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.5), sharex=True)
    
    manual_ranges_path = os.path.join(script_dir, "manual_fit_ranges.json")
    if os.path.exists(manual_ranges_path):
        with open(manual_ranges_path, "r") as f:
            manual_ranges = json.load(f)
    else:
        manual_ranges = None

    for row_idx, eps in enumerate(disorder_levels):
        data_path = find_data_file(eps)
        if data_path is None:
            print(f"Error: Data file for eps={eps} not found.")
            return

        data = np.loadtxt(data_path, delimiter=',', skiprows=1, ndmin=2)
        omegas = data[:, 0]
        
        # Dynamically determine N from file columns count
        N = data.shape[1] - 1
        current_freq_indices = [i for i in freq_indices if i < data.shape[0]]
        if not current_freq_indices:
            current_freq_indices = [0]

        col_start = 1 + N // 2
        l_abc = 100 if N >= 200 else N // 2
        col_end   = data.shape[1] - l_abc
        X = np.arange(0, col_end - col_start)

        ax_log = axes[row_idx]

        for idx, freq_idx in enumerate(current_freq_indices):
            log_env = data[freq_idx, col_start:col_end]
            amp     = np.exp(log_env)
            omega   = omegas[freq_idx]

            # Read manual ranges and check if they are available
            has_exp = False
            x_start, x_end = 5, 2800
            if manual_ranges is not None:
                key = f"{eps}_{omega:.2f}"
                if key in manual_ranges:
                    x_start, x_end = manual_ranges[key]
                    # Boundary protection
                    if x_start >= len(X):
                        x_start = 5
                    if x_end >= len(X):
                        x_end = len(X) - 5
                    mask_exp = (X >= x_start) & (X <= x_end) & (amp > 1e-10)
                    if np.sum(mask_exp) > 5:
                        p_exp = np.polyfit(X[mask_exp], np.log(amp[mask_exp]), 1)
                        gamma = -p_exp[0]
                        has_exp = True

            label = rf"$\omega = {omega:.2f}$"

            # Data curves
            ax_log.semilogy(X, np.maximum(amp, 1e-12), color=colors[idx], linewidth=2.5, alpha=0.9, label=label)

            # Fit lines and crossover markers (only drawn if manual fit was performed)
            if has_exp:
                x_exp_plot = X[(X >= x_start) & (X <= x_end)]
                y_exp_plot = np.exp(p_exp[1] - gamma * x_exp_plot)
                ax_log.semilogy(x_exp_plot, y_exp_plot, color='black', linestyle='--', linewidth=2.0, alpha=1.0, zorder=4)

                # Draw crossover point markers (larger)
                if x_start > 5:
                    ax_log.plot(x_start, amp[x_start], '+', color=colors[idx], markersize=14, markeredgewidth=2.5, zorder=10)
                if x_end < len(X) - 5:
                    ax_log.plot(x_end, amp[x_end], '+', color=colors[idx], markersize=14, markeredgewidth=2.5, zorder=10)

        # Semilogy properties
        ax_log.set_ylim(2e-13, 2.0)
        ax_log.set_xlim(0, X[-1])
        ax_log.set_yticks([1e-12, 1e-6, 1.0])
        ax_log.set_yticklabels([r"$10^{-12}$", r"$10^{-6}$", r"$10^{0}$"])
        ax_log.grid(True, which="both", alpha=0.25, linestyle='--')
        
        if row_idx == 0:
            ax_log.set_ylabel(r"$\varepsilon = 0.1$", fontweight='bold', labelpad=15)
            ax_log.set_title("Amplitude normalisée", fontweight='bold', y=1.02)
            ax_log.plot([], [], color='black', linestyle='--', linewidth=2.0, label='Ajustement')
            ax_log.plot([], [], '+', color='gray', markersize=12, markeredgewidth=2.5, label='Limites')
            ax_log.legend(bbox_to_anchor=(0.0, 1.35), loc="lower left", frameon=True, facecolor='white', framealpha=0.92, ncol=4, fontsize=11, borderaxespad=0.0)
        elif row_idx == 1:
            ax_log.set_ylabel("Amplitude de désordre\n" + r"$\varepsilon = 0.7$", fontweight='bold', labelpad=15)
        elif row_idx == 2:
            ax_log.set_ylabel(r"$\varepsilon = 1.4$", fontweight='bold', labelpad=15)

        if row_idx == 2:
            ax_log.set_xlabel(r"Distance relative $n - n_0$")
        else:
            ax_log.tick_params(labelbottom=False)

    fig.align_ylabels(axes)

    plt.tight_layout()
    save_path = os.path.join(slides_fig_dir, "08_wavepacket_envelopes_slides.pdf")
    save_path_png = os.path.join(slides_fig_dir, "08_wavepacket_envelopes_slides.png")
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_png, dpi=150, bbox_inches='tight')
    print(f"--> Generated 3-panel slides-optimized figure: {save_path} and {save_path_png}")

if __name__ == '__main__':
    plot_slides_envelopes_vertical()
