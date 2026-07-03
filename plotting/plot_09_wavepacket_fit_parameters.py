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
        filename = f"wp_envelopes_eps={eps}_N={N}{suffix}.csv"
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            return path
    
    filename = f"wp_envelopes_eps={eps}_N={N}.csv"
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

def plot_fit_parameters():
    N = 6000
    disorder_levels = [0.1, 0.7, 1.4]
    
    colors = ['#1f77b4', '#33a02c', '#d62728'] # Blue, Green, Red for diff eps

    fig, axes = plt.subplots(1, 3, figsize=(11.76, 4.66))
    
    ax_xi = axes[0]
    ax_alpha = axes[1]
    ax_nc = axes[2]
    
    for idx, eps in enumerate(disorder_levels):
        data_path = find_data_file(eps, N)
        if data_path is None:
            print(f"Error: Data file for eps={eps} not found.")
            continue
            
        data = np.loadtxt(data_path, delimiter=',', skiprows=1, ndmin=2)
        omegas = data[:, 0]
        
        col_start = 1 + N // 2
        col_end   = col_start + 2900
        X = np.arange(0, 2900)
        
        xi_list = []
        alpha_list = []
        nc_list = []
        valid_omegas = []
        
        for freq_idx in range(len(omegas)):
            log_env = data[freq_idx, col_start:col_end]
            
            best_xc = 400
            best_alpha = 0.0
            best_xi = np.nan
            best_err = float('inf')
            
            xc_range = np.arange(15, 800, 5)
            for xc in xc_range:
                mask_exp = (X >= 10) & (X <= xc)
                p_exp = np.polyfit(X[mask_exp], log_env[mask_exp], 1)
                gamma = -p_exp[0]
                xi = 1.0 / gamma if gamma > 0 else np.nan
                pred_exp = p_exp[1] - gamma * X[mask_exp]
                err_exp = np.sum((log_env[mask_exp] - pred_exp)**2)
                
                mask_pow = (X >= xc) & (X <= 2800) & (log_env > -7.8)
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
            
            xi_list.append(best_xi)
            alpha_list.append(best_alpha)
            nc_list.append(best_xc)
            valid_omegas.append(omegas[freq_idx])
            
        # Plotting
        ax_xi.semilogy(valid_omegas, xi_list, marker='o', linestyle='-', color=colors[idx], label=rf"$\varepsilon = {eps}$")
        ax_alpha.plot(valid_omegas, alpha_list, marker='s', linestyle='-', color=colors[idx], label=rf"$\varepsilon = {eps}$")
        ax_nc.plot(valid_omegas, nc_list, marker='^', linestyle='-', color=colors[idx], label=rf"$\varepsilon = {eps}$")
        
    ax_xi.set_title(r"Longueur de localisation $\xi_E$")
    ax_xi.set_xlabel(r"Fréquence $\omega$")
    ax_xi.set_ylabel(r"$\xi_E$ (échelle log)")
    ax_xi.grid(True, which="both", alpha=0.2, linestyle='--')
    
    ax_alpha.set_title(r"Exposant algébrique $\alpha$")
    ax_alpha.set_xlabel(r"Fréquence $\omega$")
    ax_alpha.set_ylabel(r"$\alpha$")
    ax_alpha.grid(True, which="both", alpha=0.2, linestyle='--')
    
    ax_nc.set_title(r"Point de transition $n_c$")
    ax_nc.set_xlabel(r"Fréquence $\omega$")
    ax_nc.set_ylabel(r"$n_c$ (sites)")
    ax_nc.grid(True, which="both", alpha=0.2, linestyle='--')
    
    ax_xi.legend()
    
    plt.tight_layout()
    save_path = os.path.join(fig_dir, "09_wavepacket_fit_parameters.pdf")
    save_path_png = os.path.join(fig_dir, "09_wavepacket_fit_parameters.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_png, dpi=150, bbox_inches='tight')
    print(f"--> Generated fit parameters figure: {save_path}")

if __name__ == '__main__':
    plot_fit_parameters()
