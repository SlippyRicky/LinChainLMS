import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress

def main():
    base_dir = "/Users/emericmellet/Desktop/Emeric/School/UPS/M1 SGM/Stage/ENSTA/Rapport/code"
    data_file = os.path.join(base_dir, "parametric_sweep", "data", "envelopes_eps=1.4_N=6000.csv")
    
    if not os.path.exists(data_file):
        print("Data file not found.")
        return
        
    data = np.loadtxt(data_file, delimiter=',', skiprows=1)
    omegas = data[:, 0]
    envelopes = data[:, 3001:6001] # aligned to X=0
    X = np.arange(0, 3000)
    
    print(f"Loaded {len(omegas)} frequencies.")
    
    # Let's test a few frequencies
    test_indices = [2, 5, 10, 15, 18]
    
    for idx in test_indices:
        omega = omegas[idx]
        y = envelopes[idx, :]
        y_norm = y / y[0]
        
        # Method 1: Amplitude threshold (where y_norm falls to 1/e)
        idx_1e = np.where(y_norm <= 1.0 / np.e)[0]
        x_c_1e = idx_1e[0] if len(idx_1e) > 0 else np.nan
        
        # Method 2: Amplitude threshold 50%
        idx_50 = np.where(y_norm <= 0.5)[0]
        x_c_50 = idx_50[0] if len(idx_50) > 0 else np.nan
        
        # Method 3: Local log-log slope threshold
        # Compute local log-log slope using a window
        valid = (X > 0) & (y_norm > 0)
        X_v = X[valid]
        y_v = y_norm[valid]
        ln_x = np.log(X_v)
        ln_y = np.log(y_v)
        
        slopes = []
        window = 10
        for i in range(len(ln_x)):
            i_min = max(0, i - window)
            i_max = min(len(ln_x), i + window)
            if i_max - i_min > 2:
                slope, _, _, _, _ = linregress(ln_x[i_min:i_max], ln_y[i_min:i_max])
                slopes.append(slope)
            else:
                slopes.append(0.0)
        slopes = np.array(slopes)
        
        # Find where local slope crosses -0.5
        idx_slope = np.where(slopes <= -0.5)[0]
        x_c_slope = X_v[idx_slope[0]] if len(idx_slope) > 0 else np.nan
        
        print(f"Omega = {omega:.4f} | x_c (1/e) = {x_c_1e} | x_c (50%) = {x_c_50} | x_c (slope=-0.5) = {x_c_slope}")

if __name__ == "__main__":
    main()
