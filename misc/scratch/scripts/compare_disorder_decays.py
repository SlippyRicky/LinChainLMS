import numpy as np
import os

def main():
    base_dir = "/Users/emericmellet/Desktop/Emeric/School/UPS/M1 SGM/Stage/ENSTA/Rapport/code"
    
    for eps in [0.1, 0.7, 1.4]:
        f = os.path.join(base_dir, "parametric_sweep", "data", f"envelopes_eps={eps}_N={N}.csv" if 'N' in locals() else f"envelopes_eps={eps}_N=6000.csv")
        if os.path.exists(f):
            data = np.loadtxt(f, delimiter=',', skiprows=1)
            omegas = data[:, 0]
            envelopes = data[:, 3001:6001]
            
            # Print for a medium frequency (e.g. idx = 5, omega around 0.76)
            idx = 5
            y = envelopes[idx, :]
            y_norm = y / y[0]
            print(f"Disorder eps = {eps} | Freq = {omegas[idx]:.4f} | y(0) = {y[0]:.4f} | y(1000) = {y[1000]:.4f} (norm={y_norm[1000]:.4f})")

if __name__ == "__main__":
    main()
