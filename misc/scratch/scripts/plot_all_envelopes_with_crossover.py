import numpy as np
import matplotlib.pyplot as plt
import os

def find_data_file(eps, N, test=False):
    """Recherche le fichier de données dans plusieurs répertoires possibles."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    suffix = "_test" if test else ""
    filename = f"envelopes_eps={eps}_N={N}{suffix}.csv"
    
    paths_to_check = [
        os.path.join(project_root, "parametric_sweep", "data", filename),
        os.path.join(project_root, "scratch", "data", filename),
        os.path.join(project_root, "code", "parametric_sweep", "data", filename),
        os.path.join(project_root, "code", "scratch", filename),
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "..", "data", filename)
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            return path
            
    return None

def plot_disorder_envelopes(eps, N):
    data_file = find_data_file(eps, N, test=False)
    if data_file is None:
        data_file = find_data_file(eps, N, test=True)
        
    if data_file is None:
        print(f"Avertissement : Fichier de données introuvable pour eps={eps}, N={N}")
        return False
        
    print(f"Lecture du fichier {os.path.basename(data_file)}...")
    try:
        data = np.loadtxt(data_file, delimiter=',', skiprows=1)
    except Exception as e:
        print(f"Erreur de lecture : {e}")
        return False
        
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "lines.linewidth": 1.0,
        "grid.alpha": 0.3
    })
    
    col_start = 1 + N // 2
    col_end = col_start + N // 2
    
    step = 100
    envelopes = data[1::step, col_start:col_end]
    if envelopes.shape[0] == 0:
        envelopes = data[:, col_start:col_end]
        
    Y = envelopes.T
    X = np.arange(0, N // 2)
    
    # Calcul des points de crossover via Curvature Peak
    from scipy.signal import savgol_filter
    x_c_vals = []
    y_c_vals = []
    trunc_idx = 2500
    
    for j in range(envelopes.shape[0]):
        y = envelopes[j, :]
        X_t = X[1:trunc_idx]
        y_t = y[1:trunc_idx]
        
        u = np.log(X_t)
        v = np.log(y_t)
        
        v_smooth = savgol_filter(v, window_length=51, polyorder=3)
        
        du = np.gradient(u)
        dv = np.gradient(v_smooth, u)
        d2v = np.gradient(dv, u)
        
        curvature = np.abs(d2v) / (1 + dv**2)**1.5
        
        peak_idx = np.argmax(curvature)
        
        x_c_vals.append(X_t[peak_idx])
        y_c_vals.append(y_t[peak_idx])
            
    x_c_vals = np.array(x_c_vals)
    y_c_vals = np.array(y_c_vals)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Enveloppes spatiales avec Crossover Knees - Désordre $\\epsilon = {eps}$ (N = {N})", fontsize=16)
    
    # --- Tracé 1 : Linéaire / Linéaire ---
    ax = axes[0]
    ax.plot(X, Y)
    ax.plot(x_c_vals, y_c_vals, 'ko', markersize=5, zorder=10, label="True Crossover Knee")
    ax.grid(True)
    ax.set_title("Échelle Linéaire / Linéaire")
    ax.set_xlabel("Distance depuis le nœud d'excitation $x$")
    ax.set_ylabel("Amplitude de l'enveloppe $|u(x)|$")
    ax.legend(loc='upper right')
    
    # --- Tracé 2 : Linéaire / Log ---
    ax = axes[1]
    ax.semilogy(X, Y)
    ax.semilogy(x_c_vals, y_c_vals, 'ko', markersize=5, zorder=10)
    ax.grid(True, which="both")
    ax.set_title("Échelle Linéaire / Log")
    ax.set_xlabel("Distance depuis le nœud d'excitation $x$")
    ax.set_ylabel("Amplitude $|u(x)|$ (Log)")
    ax.set_ylim(1e-5, max(1.0, np.max(Y) * 1.1))
    
    # --- Tracé 3 : Log / Log ---
    ax = axes[2]
    ax.loglog(X[1:], Y[1:, :])
    valid = ~np.isnan(x_c_vals) & (x_c_vals > 0)
    ax.loglog(x_c_vals[valid], y_c_vals[valid], 'ko', markersize=5, zorder=10)
    ax.grid(True, which="both")
    ax.set_title("Échelle Log / Log")
    ax.set_xlabel("Distance depuis le nœud d'excitation $x$ (Log)")
    ax.set_ylabel("Amplitude $|u(x)|$ (Log)")
    ax.set_ylim(1e-5, max(1.0, np.max(Y) * 1.1))
    
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_png = os.path.join(script_dir, f"envelopes_with_crossover_eps={eps}_N={N}.png")
    plt.savefig(output_png, dpi=300)
    plt.close()
    
    print(f"Graphiques générés pour eps={eps} : {output_png}")
    return True

def main():
    disorder_levels = [0.1, 0.7, 1.4]
    N = 6000
    for eps in disorder_levels:
        plot_disorder_envelopes(eps, N)

if __name__ == "__main__":
    main()
