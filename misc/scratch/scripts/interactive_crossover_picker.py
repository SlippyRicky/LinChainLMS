import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def find_data_file(eps, N):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    filename = f"envelopes_eps={eps}_N={N}.csv"
    paths_to_check = [
        os.path.join(project_root, "parametric_sweep", "data", filename),
        os.path.join(project_root, "code", "parametric_sweep", "data", filename),
        os.path.join(project_root, "code", "scratch", filename),
        os.path.join(script_dir, filename)
    ]
    for path in paths_to_check:
        if os.path.exists(path): return path
    return None

def main():
    N = 6000
    disorder_levels = [0.1, 0.7, 1.4]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_out = os.path.join(script_dir, "manual_crossover_points.csv")
    
    if os.path.exists(csv_out):
        df = pd.read_csv(csv_out)
        saved_points = {(row['eps'], row['omega']): (row['x_c'], row['y_c']) for _, row in df.iterrows()}
        print(f"Loaded {len(saved_points)} existing points. Picking up where you left off!")
    else:
        saved_points = {}
        
    records = []
    
    for eps in disorder_levels:
        data_file = find_data_file(eps, N)
        if not data_file: 
            print(f"Data file not found for eps={eps}")
            continue
            
        data = np.loadtxt(data_file, delimiter=',', skiprows=1)
        omegas = data[:, 0]
        envelopes = data[:, 1 + N//2 : 1 + N]
        X = np.arange(0, N//2)
        
        for i, omega in enumerate(omegas):
            if (eps, omega) in saved_points:
                x_c, y_c = saved_points[(eps, omega)]
                records.append({'eps': eps, 'omega': omega, 'x_c': x_c, 'y_c': y_c})
                continue
                
            y = envelopes[i, :]
            trunc_idx = 2800
            X_t = X[1:trunc_idx]
            y_t = y[1:trunc_idx]
            
            from scipy.signal import savgol_filter
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
            
            # --- Top Plot: Envelope ---
            ax1.loglog(X_t, y_t, label=f"Disorder $\epsilon={eps}$, $\omega={omega:.3f}$ (Raw)", alpha=0.5)
            
            # Smooth the data heavily to guide the eye and prevent derivative noise amplification
            u = np.log(X_t)
            v = np.log(y_t)
            v_smooth = savgol_filter(v, window_length=201, polyorder=3)
            ax1.loglog(X_t, np.exp(v_smooth), 'r--', label="Heavily Smoothed Envelope", linewidth=2)
            
            ax1.set_title(f"CLICK ON THE TOP PLOT to select the Knee\nProgress: {len(records)}/60", fontsize=14)
            ax1.set_ylabel("Amplitude $|u(x)|$")
            ax1.set_ylim(1e-6, 1e2)
            ax1.grid(True, which="both")
            ax1.legend(loc="upper right")
            
            # --- Bottom Plot: Curvature ---
            du = np.gradient(u)
            dv = np.gradient(v_smooth, u)
            d2v = np.gradient(dv, u)
            raw_curvature = np.abs(d2v) / (1 + dv**2)**1.5
            
            # Smooth the curvature itself to hide the remaining noise spikes
            curvature_smooth = savgol_filter(raw_curvature, window_length=201, polyorder=3)
            
            ax2.plot(X_t, curvature_smooth, color='purple', label="Smoothed Curvature", linewidth=2)
            ax2.set_xscale('log')
            ax2.set_xlabel("Distance $x$")
            ax2.set_ylabel("Log-Log Curvature")
            ax2.grid(True, which="both")
            ax2.legend(loc="upper right")
            
            plt.tight_layout()
            
            print(f"Waiting for click... (eps={eps}, omega={omega:.3f})")
            point = plt.ginput(1, timeout=-1)
            plt.close(fig)
            
            if point:
                x_c, y_c = point[0]
                records.append({'eps': eps, 'omega': omega, 'x_c': x_c, 'y_c': y_c})
                pd.DataFrame(records).to_csv(csv_out, index=False)
            else:
                print("Skipped or canceled. Exiting tool.")
                pd.DataFrame(records).to_csv(csv_out, index=False)
                return
                
    print(f"\nAwesome! You've selected all {len(records)} points.")
    print(f"Data saved to {csv_out}.")

if __name__ == "__main__":
    main()
