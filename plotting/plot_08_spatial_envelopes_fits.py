import matplotlib.pyplot as plt
import numpy as np
import os

# --- STYLE CONFIGURATION (Premium Manuscript) ---
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 21,
    "legend.fontsize": 17,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.2
})

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../../"))
import plot_config
fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
# Directory creation handled by plot_config

def find_data_file(eps, N):
    data_dir = plot_config.DATA_DIR
    
    for suffix in ["", "_test"]:
        filename = f"envelopes_eps={eps}_N={N}{suffix}.csv"
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return path
            
    # Fallback to old paths
    filename = f"envelopes_eps={eps}_N={N}.csv"
    paths_to_check = [
        os.path.join(project_root, "code", "parametric_sweep", "data", filename),
        os.path.join(project_root, "code", "scratch", "data", filename),
        os.path.join(project_root, "parametric_sweep", "data", filename),
        os.path.join(project_root, "scratch", "data", filename),
    ]
    for path in paths_to_check:
        if os.path.exists(path):
            return path
    return None

def plot_all_envelopes():
    N = 6000
    disorder_levels = [0.1, 0.7, 1.4]
    freq_indices = [1, 5, 10, 15, 19] # 5 representative frequencies

    # Premium color palette: blue (low freq) to purple (high freq)
    colors = ['#1f77b4', '#33a02c', '#ff7f0e', '#e31a1c', '#6a3d9a']

    fig, axes = plt.subplots(3, 2, figsize=(11.76, 14.69))

    for row_idx, eps in enumerate(disorder_levels):
        data_path = find_data_file(eps, N)
        if data_path is None:
            print(f"Error: Data file for eps={eps} not found.")
            return

        data = np.loadtxt(data_path, delimiter=',', skiprows=1, ndmin=2)
        omegas = data[:, 0]

        current_freq_indices = [i for i in freq_indices if i < data.shape[0]]
        if not current_freq_indices:
            current_freq_indices = [0]

        col_start = 1 + N // 2
        col_end   = col_start + 2900
        X = np.arange(0, 2900)

        ax_log    = axes[row_idx, 0]  # normalised amplitude (semilogy)
        ax_loglog = axes[row_idx, 1]  # normalised amplitude (loglog)

        for idx, freq_idx in enumerate(current_freq_indices):
            raw_env = data[freq_idx, col_start:col_end]
            amp = raw_env / raw_env[0]
            log_env = np.log(np.maximum(amp, 1e-15))
            omega   = omegas[freq_idx]

            # Sweep crossover point x_c to find the best fit
            best_xc = 400
            best_alpha = 0.0
            best_xi = np.nan
            best_err = float('inf')
            
            xc_range = np.arange(15, 800, 5)
            for xc in xc_range:
                # Exponential fit on [10, xc]
                mask_exp = (X >= 10) & (X <= xc)
                p_exp = np.polyfit(X[mask_exp], log_env[mask_exp], 1)
                gamma = -p_exp[0]
                xi = 1.0 / gamma if gamma > 0 else np.nan
                pred_exp = p_exp[1] - gamma * X[mask_exp]
                err_exp = np.sum((log_env[mask_exp] - pred_exp)**2)
                
                # Power law fit on [xc, 2000]
                mask_pow = (X >= xc) & (X <= 2000) & (log_env > -7.8)
                if np.sum(mask_pow) > 10:
                    p_pow = np.polyfit(np.log(X[mask_pow]), log_env[mask_pow], 1)
                    alpha = -p_pow[0]
                    pred_pow = p_pow[1] - alpha * np.log(X[mask_pow])
                    err_pow = np.sum((log_env[mask_pow] - pred_pow)**2)
                else:
                    alpha = np.nan
                    err_pow = 0
                    
                total_err = err_exp + err_pow
                if total_err < best_err:
                    best_err = total_err
                    best_xc = xc
                    best_alpha = alpha
                    best_xi = xi
            
            x_c = best_xc
            xi = best_xi
            alpha = best_alpha

            # Re-generate prediction lines for plotting
            p_exp = np.polyfit(X[(X >= 10) & (X <= x_c)], log_env[(X >= 10) & (X <= x_c)], 1)
            
            mask_pow_fit = (X >= x_c) & (X <= 2000) & (log_env > -7.8)
            if np.sum(mask_pow_fit) > 5 and not np.isnan(alpha):
                p_pow = np.polyfit(np.log(X[mask_pow_fit]), log_env[mask_pow_fit], 1)
                has_pow = True
            else:
                has_pow = False

            # Label: frequency only
            label = rf"$\omega = {omega:.2f}$"

            # --- DATA CURVES (drawn first, underneath fits) ---
            ax_log.semilogy(X, np.maximum(amp, 1e-12), color=colors[idx], linewidth=1.5, alpha=0.85, label=label)
            ax_loglog.loglog(X[1:], np.maximum(amp[1:], 1e-12), color=colors[idx], linewidth=1.5, alpha=0.85, label=label)

            # --- FIT LINES ---
            x_exp_plot = X[(X >= 10) & (X <= x_c)]
            y_exp_plot = np.exp(p_exp[1] - (1.0 / xi) * x_exp_plot) if not np.isnan(xi) else np.ones_like(x_exp_plot)
            ax_log.semilogy(x_exp_plot, y_exp_plot,
                            color='black', linestyle='--', linewidth=1.5,
                            alpha=1.0, zorder=4)

            if has_pow:
                x_pow_plot = X[(X >= max(x_c, 2)) & (X <= 2000)]
                y_pow_plot = np.exp(p_pow[1] - alpha * np.log(x_pow_plot))
                ax_loglog.loglog(x_pow_plot, y_pow_plot,
                                 color='black', linestyle=':', linewidth=1.5,
                                 alpha=1.0, zorder=4)

            # Draw Crossover Point Marker on each curve (both plots)
            ax_log.scatter(x_c, amp[x_c], color=colors[idx], marker='o', s=70, edgecolor='black', linewidths=1.5, zorder=5)
            ax_loglog.scatter(x_c, amp[x_c], color=colors[idx], marker='o', s=70, edgecolor='black', linewidths=1.5, zorder=5)


        # Semilogy properties
        ax_log.set_ylim(1e-5, 2.0)
        ax_log.set_xlim(0, 2900)
        ax_log.grid(True, which="both", alpha=0.2, linestyle='--')
        ax_log.set_ylabel(
            rf"Amplitude $\langle|u_n/u_0|\rangle_M$  ($\varepsilon = {eps}$)",
            fontweight='bold')

        # Log-Log properties
        ax_loglog.set_ylim(1e-5, 2.0)
        ax_loglog.set_xlim(1, 2900)
        ax_loglog.grid(True, which="both", alpha=0.2, linestyle='--')

        # Titles for the top row
        if row_idx == 0:
            ax_log.set_title("Amplitude normalisée (semi-log)", fontweight='bold')
            ax_loglog.set_title("Amplitude normalisée (log-log)", fontweight='bold')

        # Legend: ONLY on the top-row subplots (row 0, columns 0 and 1), at bottom left
        if row_idx == 0:
            ax_log.plot([], [], color='black', linestyle='--', linewidth=1.5, label='Ajustement exponentiel')
            ax_log.legend(loc='lower left', frameon=True, facecolor='white',
                          framealpha=0.92, ncol=1)

            ax_loglog.plot([], [], color='black', linestyle=':', linewidth=1.5, label='Ajustement loi de puissance')
            ax_loglog.legend(loc='lower left', frameon=True, facecolor='white',
                             framealpha=0.92, ncol=1)

        # X-labels for the bottom row only
        if row_idx == 2:
            ax_log.set_xlabel(r"Distance relative $n - n_0$")
            ax_loglog.set_xlabel(r"Distance relative $n - n_0$")
        else:
            ax_log.tick_params(labelbottom=False)
            ax_loglog.tick_params(labelbottom=False)

    plt.tight_layout()
    save_path = os.path.join(fig_dir, "08_spatial_envelopes.pdf")
    save_path_png = os.path.join(fig_dir, "08_spatial_envelopes.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_png, dpi=150, bbox_inches='tight')
    print(f"--> Generated ensemble-averaged spatial envelope figure: {save_path} and {save_path_png}")

if __name__ == '__main__':
    plot_all_envelopes()
