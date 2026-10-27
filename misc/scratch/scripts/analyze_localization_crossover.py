import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
from scipy.signal import savgol_filter

def find_data_file(eps, N, test=False):
    """Recherche le fichier de données dans plusieurs répertoires possibles."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    suffix = "_test" if test else ""
    filename = f"envelopes_eps={eps}_N={N}{suffix}.csv"
    
    paths_to_check = [
        os.path.join(project_root, "parametric_sweep", "data", filename),
        os.path.join(project_root, "scratch", filename),
        os.path.join(project_root, "scratch", "Analyse", "data", filename),
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "Analyse", "data", filename),
        os.path.join(script_dir, "..", "data", filename)
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            return path
            
    return None

def analyze_disorder(eps, N):
    data_file = find_data_file(eps, N, test=False)
    if data_file is None:
        data_file = find_data_file(eps, N, test=True)
        
    if data_file is None:
        print(f"Fichier de données introuvable pour eps={eps}")
        return None
        
    print(f"Analyse de {os.path.basename(data_file)}...")
    data = np.loadtxt(data_file, delimiter=',', skiprows=1)
    
    omegas = data[:, 0]
    # Slicing alignant le nœud d'excitation u_3000 à X = 0
    col_start = 1 + N // 2
    col_end = col_start + N // 2
    envelopes = data[:, col_start:col_end]
    
    X = np.arange(0, N // 2)
    
    # 1. Ajustement de L(omega) pour chaque fréquence (sur envelopes normalisées)
    L_vals = []
    r2_L_vals = []
    
    # Fenêtre d'ajustement exponentiel : X de 300 à 2200 sites
    fit_mask = (X >= 300) & (X <= 2200)
    X_fit = X[fit_mask]
    
    for i in range(len(omegas)):
        y = envelopes[i, :]
        y_source = y[0]
        if y_source <= 0:
            L_vals.append(np.nan)
            r2_L_vals.append(0.0)
            continue
            
        y_norm = y / y_source
        y_fit = y_norm[fit_mask]
        
        valid = y_fit > 1e-12
        if np.sum(valid) < 10:
            L_vals.append(np.nan)
            r2_L_vals.append(0.0)
            continue
            
        slope, intercept, r_val, p_val, std_err = linregress(X_fit[valid], np.log(y_fit[valid]))
        
        if slope < 0:
            L = -1.0 / slope
            L_vals.append(L)
            r2_L_vals.append(r_val**2)
        else:
            L_vals.append(np.nan)
            r2_L_vals.append(0.0)
            
    L_vals = np.array(L_vals)
    r2_L_vals = np.array(r2_L_vals)
    
    # We will truncate the data to remove the effect of Periodic Boundary Conditions (PBC)
    # The chain length from the source to the boundary is N/2 = 3000. 
    # We truncate the last 500 points (X > 2500) to ensure the tail rise doesn't confuse the algorithm.
    x_c_vals = []
    y_c_vals = []
    trunc_idx = 2500
    
    for i in range(len(omegas)):
        y = envelopes[i, :]
        
        # Avoid the very first point (X=0) for log scales
        X_t = X[1:trunc_idx]
        y_t = y[1:trunc_idx]
        
        u = np.log(X_t)
        v = np.log(y_t)
        
        # Smooth the log-log data to compute stable derivatives
        v_smooth = savgol_filter(v, window_length=51, polyorder=3)
        
        # Calculate derivatives
        du = np.gradient(u)
        dv = np.gradient(v_smooth, u)
        d2v = np.gradient(dv, u)
        
        # Curvature formula
        curvature = np.abs(d2v) / (1 + dv**2)**1.5
        
        # Find the peak of curvature which marks the transition from exponential to inverse power-law
        peak_idx = np.argmax(curvature)
        
        # Map back to real coordinates
        x_c = X_t[peak_idx]
        y_c = y_t[peak_idx]
        
        x_c_vals.append(x_c)
        y_c_vals.append(y_c)
        
    x_c_vals = np.array(x_c_vals)
    y_c_vals = np.array(y_c_vals)
    
    return {
        "omegas": omegas,
        "envelopes": envelopes,
        "X": X,
        "L_vals": L_vals,
        "r2_L_vals": r2_L_vals,
        "x_c_vals": x_c_vals,
        "y_c_vals": y_c_vals
    }

def main():
    disorder_levels = [0.1, 0.7, 1.4]
    N = 6000
    
    results = {}
    for eps in disorder_levels:
        res = analyze_disorder(eps, N)
        if res is not None:
            results[eps] = res
            
    if len(results) == 0:
        print("Aucune donnée chargée, arrêt.")
        return
        
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "grid.alpha": 0.3
    })
    
    colors = {0.1: "#1f77b4", 0.7: "#2ca02c", 1.4: "#d62728"}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # =========================================================================
    # FIGURE 1 : LONGUEUR DE LOCALISATION L(omega) EN ESPACE LINÉAIRE
    # =========================================================================
    plt.figure(figsize=(9, 7))
    for eps in disorder_levels:
        if eps not in results:
            continue
        res = results[eps]
        omega_v = res["omegas"]
        L_v = res["L_vals"]
        
        # Filtre des valeurs valides
        valid = ~np.isnan(L_v) & (L_v > 0)
        
        plt.plot(omega_v[valid], L_v[valid], 'o-', color=colors[eps], linewidth=1.5, markersize=6, label=f"$\\epsilon = {eps}$")
        
    plt.grid(True, alpha=0.5)
    plt.xlabel("Fréquence d'excitation $\\omega$")
    plt.ylabel("Longueur de localisation $L(\\omega)$ (mailles)")
    plt.title("Longueur de localisation $L(\\omega)$ en espace linéaire")
    plt.legend()
    plt.tight_layout()
    
    output_L_lin = os.path.join(script_dir, "combined_localization_lengths_linear.png")
    plt.savefig(output_L_lin, dpi=300)
    plt.close()
    print(f"Figure combinée L(omega) linéaire : {output_L_lin}")
    
    # =========================================================================
    # FIGURE 2 : KNEE CROSSOVER TRAJECTORY (UNNORMALIZED LOG-LOG)
    # =========================================================================
    plt.figure(figsize=(10, 8))
    
    # Afin de voir les enveloppes brutes en arrière-plan, on trace celles de eps=1.4
    if 1.4 in results:
        res_14 = results[1.4]
        X = res_14["X"]
        for j in range(len(res_14["omegas"])):
            plt.loglog(X[1:], res_14["envelopes"][j, 1:], color='gray', alpha=0.25, linewidth=0.6)
            
    for eps in disorder_levels:
        if eps not in results:
            continue
        res = results[eps]
        x_c = res["x_c_vals"]
        y_c = res["y_c_vals"]
        
        valid = ~np.isnan(x_c) & ~np.isnan(y_c) & (x_c > 0) & (y_c > 0)
        x_c_v = x_c[valid]
        y_c_v = y_c[valid]
        
        plt.plot(x_c_v, y_c_v, 'o', color=colors[eps], markersize=6, label=f"Knees $\\epsilon = {eps}$")
        
        # Ajustement de la trajectoire en log-log : ln(y_c) = A * ln(x_c) + B
        if len(x_c_v) > 2:
            slope, intercept, r_val, p_val, std_err = linregress(np.log(x_c_v), np.log(y_c_v))
            x_fit = np.logspace(np.log10(np.min(x_c_v)), np.log10(np.max(x_c_v)), 100)
            y_fit = np.exp(intercept) * x_fit**slope
            plt.loglog(x_fit, y_fit, '--', color=colors[eps], 
                       label=f"Fit $\\epsilon = {eps}$ ($Y \\propto X^{{{slope:.2f}}}$, $R^2 = {r_val**2:.3f}$)")
            print(f"[{eps}] Trajectoire du Knee : Y_c = {np.exp(intercept):.3f} * X_c^({slope:.4f}) (R2 = {r_val**2:.4f})")
            
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Distance x (Log)")
    plt.ylabel("Amplitude de l'enveloppe brute Y (Log)")
    plt.title("Trajectoire du Crossover Knee (1/e) pour différents désordres")
    plt.legend()
    plt.xlim(1, 3000)
    plt.ylim(1e-4, 100.0)
    plt.tight_layout()
    
    output_crossover = os.path.join(script_dir, "combined_crossover_points.png")
    plt.savefig(output_crossover, dpi=300)
    plt.close()
    print(f"Figure combinée crossover points : {output_crossover}")
    
    # =========================================================================
    # FIGURE 3-6 : CROSSOVER KNEE vs FREQUENCY (POWER-LAW FIT)
    # =========================================================================
    # Preparer les donnees de fit
    fit_data = {}
    for eps in disorder_levels:
        if eps not in results: continue
        res = results[eps]
        x_c_vals = res["x_c_vals"]
        
        # Filtre d'artefacts aux extremes (saturation et plancher)
        valid = ~np.isnan(x_c_vals) & (x_c_vals > 10) & (x_c_vals < 2800)
        
        if np.sum(valid) > 2:
            omega_v = res["omegas"][valid]
            x_c_v = x_c_vals[valid]
            
            # Power-Law fit: ln(x_c) = ln(A) - B * ln(omega)
            slope, intercept, r_val, p_val, std_err = linregress(np.log(omega_v), np.log(x_c_v))
            A = np.exp(intercept)
            B = -slope
            
            fit_data[eps] = {'omega_v': omega_v, 'x_c_v': x_c_v, 'A': A, 'B': B, 'R2': r_val**2}
            print(f"[{eps}] Ajustement Loi de Puissance : x_c = {A:.1f} * omega^(-{B:.3f}) (R2 = {r_val**2:.4f})")

    # Helper pour tracer les 4 variantes
    def plot_xc_vs_omega_variants(x_scale, y_scale, swap_axes, filename, title):
        plt.figure(figsize=(9, 7))
        for eps in disorder_levels:
            if eps not in fit_data: continue
            fd = fit_data[eps]
            
            omega_fit = np.logspace(np.log10(np.min(fd['omega_v'])), np.log10(np.max(fd['omega_v'])), 100)
            x_fit = fd['A'] * omega_fit**(-fd['B'])
            
            label_data = f"Knees $\\epsilon = {eps}$"
            label_fit = f"Fit $\\epsilon = {eps}$ ($x_c \\propto \\omega^{{{-fd['B']:.2f}}}$, $R^2 = {fd['R2']:.3f}$)"
            
            if swap_axes:
                plt.plot(fd['x_c_v'], fd['omega_v'], 's', color=colors[eps], markersize=6, label=label_data)
                plt.plot(x_fit, omega_fit, '-', color=colors[eps], label=label_fit)
            else:
                plt.plot(fd['omega_v'], fd['x_c_v'], 's', color=colors[eps], markersize=6, label=label_data)
                plt.plot(omega_fit, x_fit, '-', color=colors[eps], label=label_fit)

        plt.grid(True, which="both", alpha=0.5)
        
        if swap_axes:
            plt.xlabel("Position du Knee $x_c$ (mailles)")
            plt.ylabel("Fréquence d'excitation $\\omega$")
            if x_scale == 'log': plt.xscale('log')
            if y_scale == 'log': plt.yscale('log')
        else:
            plt.xlabel("Fréquence d'excitation $\\omega$")
            plt.ylabel("Position du Knee $x_c$ (mailles)")
            if x_scale == 'log': plt.xscale('log')
            if y_scale == 'log': plt.yscale('log')

        plt.title(title)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(script_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Figure sauvegardee : {out_path}")

    plot_xc_vs_omega_variants('linear', 'linear', False, "combined_crossover_vs_omega_powerlaw_linear.png", "Position $x_c$ vs Fréquence $\\omega$ (Linéaire)")
    plot_xc_vs_omega_variants('log', 'log', False, "combined_crossover_vs_omega_powerlaw_loglog.png", "Position $x_c$ vs Fréquence $\\omega$ (Echelle Log-Log)")
    plot_xc_vs_omega_variants('linear', 'linear', True, "combined_omega_vs_crossover_powerlaw_linear.png", "Fréquence $\\omega$ vs Position $x_c$ (Linéaire)")
    plot_xc_vs_omega_variants('log', 'log', True, "combined_omega_vs_crossover_powerlaw_loglog.png", "Fréquence $\\omega$ vs Position $x_c$ (Echelle Log-Log)")

    # =========================================================================
    # FIGURE 3 : SELECTION D'ENVELOPPES BRUTES COMPARÉES
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Comparaison des enveloppes brutes à différentes fréquences", fontsize=16)
    
    # Selection de 3 fréquences cibles: basse, moyenne et haute
    target_omegas = [0.313, 0.908, 1.521]
    
    for k, target in enumerate(target_omegas):
        ax = axes[k]
        
        for eps in disorder_levels:
            if eps not in results:
                continue
            res = results[eps]
            X = res["X"]
            
            # Trouver l'indice de la fréquence la plus proche de la cible
            idx_omega = np.argmin(np.abs(res["omegas"] - target))
            actual_omega = res["omegas"][idx_omega]
            y = res["envelopes"][idx_omega, :]
            
            ax.loglog(X[1:], y[1:], color=colors[eps], linewidth=1.2, label=f"$\\epsilon = {eps}$")
            
        ax.set_title(f"Fréquence $\\omega \\approx {target}$ (réelle {actual_omega:.3f})")
        ax.set_xlabel("Distance $x$ (Log)")
        ax.grid(True, which="both", alpha=0.3)
        if k == 0:
            ax.set_ylabel("Amplitude de l'enveloppe brute Y (Log)")
            ax.legend()
            
    plt.ylim(1e-4, 100.0)
    plt.tight_layout()
    
    output_selection = os.path.join(script_dir, "combined_multiscale_envelopes_selection.png")
    plt.savefig(output_selection, dpi=300)
    plt.close()
    print(f"Figure combinée sélection enveloppes : {output_selection}")

if __name__ == "__main__":
    main()
