import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import json
import os

import plot_config
script_dir = os.path.dirname(os.path.abspath(__file__))

def find_data_file(eps, N):
    data_dir = plot_config.DATA_DIR
    for suffix in ["", "_test"]:
        filename = f"wp_envelopes_eps={eps}_N={N}{suffix}.csv"
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return path
    return None

manual_ranges = {}
output_file = "manual_fit_ranges.json"

# Collect all datasets
N = 6000
disorder_levels = [0.1, 0.7, 1.4]
freq_indices = [1, 5, 10, 15, 19]

datasets = []
for eps in disorder_levels:
    data_path = find_data_file(eps, N)
    if data_path:
        data = np.loadtxt(data_path, delimiter=',', skiprows=1, ndmin=2)
        omegas = data[:, 0]
        current_freq_indices = [i for i in freq_indices if i < data.shape[0]]
        if not current_freq_indices:
            current_freq_indices = [0]
        for freq_idx in current_freq_indices:
            datasets.append((eps, omegas[freq_idx], data_path, freq_idx))

if not datasets:
    print(f"No datasets found!")
    exit(1)

def process_dataset(idx):
    if idx >= len(datasets):
        with open(output_file, "w") as f:
            json.dump(manual_ranges, f, indent=4)
        print(f"\n✅ All done! Saved your ranges to {output_file}")
        plt.close('all')
        return

    eps, omega, data_path, freq_idx = datasets[idx]
    
    data = np.loadtxt(data_path, delimiter=',', skiprows=1, ndmin=2)
    col_start = 1 + N // 2
    col_end   = col_start + 2900
    X = np.arange(0, 2900)
    log_env = data[freq_idx, col_start:col_end]
    amp     = np.exp(log_env)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2)
    
    line_data, = ax.semilogy(X, np.maximum(amp, 1e-12), label='Data', color='blue')
    line_fit, = ax.semilogy([], [], color='red', linewidth=2, label='Fit')
    vline_start = ax.axvline(-100, color='k', linestyle=':')
    vline_end = ax.axvline(-100, color='k', linestyle=':')
    valid_amp = amp[amp > 1e-12]
    min_amp = np.min(valid_amp) if len(valid_amp) > 0 else 1e-10
    
    ax.set_xlim(0, max(X))
    ax.set_ylim(min_amp / 2, max(amp) * 2)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')
    
    title_text = f"Dataset {idx+1}/{len(datasets)}: Eps={eps}, Omega={omega:.2f}\nClick 2 points on the plot (Start, then End)."
    ax.set_title(title_text)

    # State variables
    pts = []
    
    def on_click(event):
        if event.inaxes != ax: return
        if event.button != 1: return
        
        pts.append(event.xdata)
        if len(pts) == 1:
            vline_start.set_xdata([pts[0], pts[0]])
        elif len(pts) == 2:
            vline_end.set_xdata([pts[1], pts[1]])
            x_start, x_end = sorted(pts)
            mask = (X >= x_start) & (X <= x_end) & (amp > 1e-10)
            if np.sum(mask) > 1:
                p = np.polyfit(X[mask], np.log(amp[mask]), 1)
                gamma = -p[0]
                y_fit = np.exp(p[1] - gamma * X[mask])
                line_fit.set_data(X[mask], y_fit)
            ax.set_title(f"Dataset {idx+1}/{len(datasets)}: Eps={eps}, Omega={omega:.2f}\nFit shown. Click 'Confirm' to proceed, or 'Retry' to clear.")
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)

    # Buttons
    ax_confirm = plt.axes([0.7, 0.05, 0.2, 0.075])
    btn_confirm = Button(ax_confirm, 'Confirm', color='lightgreen', hovercolor='palegreen')
    
    ax_retry = plt.axes([0.45, 0.05, 0.2, 0.075])
    btn_retry = Button(ax_retry, 'Retry', color='lightcoral', hovercolor='salmon')

    ax_full = plt.axes([0.2, 0.05, 0.2, 0.075])
    btn_full = Button(ax_full, 'Full Dataset', color='lightblue', hovercolor='skyblue')

    def confirm(event):
        if len(pts) >= 2:
            x_start, x_end = sorted(pts[:2])
            manual_ranges[f"{eps}_{omega:.2f}"] = (int(x_start), int(x_end))
            plt.close(fig)
            process_dataset(idx + 1)
        else:
            ax.set_title("Need exactly 2 points before confirming!\n" + title_text)
            fig.canvas.draw()

    def retry(event):
        pts.clear()
        vline_start.set_xdata([-100, -100])
        vline_end.set_xdata([-100, -100])
        line_fit.set_data([], [])
        ax.set_title(title_text)
        fig.canvas.draw()

    def full_dataset(event):
        pts.clear()
        pts.extend([5, 2800])
        vline_start.set_xdata([5, 5])
        vline_end.set_xdata([2800, 2800])
        x_start, x_end = 5, 2800
        mask = (X >= x_start) & (X <= x_end) & (amp > 1e-10)
        if np.sum(mask) > 1:
            p = np.polyfit(X[mask], np.log(amp[mask]), 1)
            gamma = -p[0]
            y_fit = np.exp(p[1] - gamma * X[mask])
            line_fit.set_data(X[mask], y_fit)
        ax.set_title(f"Dataset {idx+1}/{len(datasets)}: Eps={eps}, Omega={omega:.2f}\nFull dataset fit shown. Click 'Confirm' to proceed, or 'Retry'.")
        fig.canvas.draw()

    btn_confirm.on_clicked(confirm)
    btn_retry.on_clicked(retry)
    btn_full.on_clicked(full_dataset)
    
    plt.show()

# Start the interactive loop
process_dataset(0)
