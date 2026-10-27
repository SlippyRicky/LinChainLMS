import numpy as np
import os
import matplotlib.pyplot as plt

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
    
    plt.figure(figsize=(10, 8))
    
    # Plot all envelopes in log-log
    for i in range(len(omegas)):
        y = envelopes[i, :]
        y_norm = y / y[0]
        plt.loglog(X[1:], y_norm[1:], color='gray', alpha=0.3)
        
    x_c_1e = []
    y_c_1e = []
    
    x_c_01 = []
    y_c_01 = []
    
    x_c_slope_05 = []
    y_c_slope_05 = []
    
    x_c_slope_1 = []
    y_c_slope_1 = []
    
    for i in range(len(omegas)):
        y = envelopes[i, :]
        y_norm = y / y[0]
        
        # 1. 1/e threshold
        idx_1e = np.where(y_norm <= 1.0 / np.e)[0]
        if len(idx_1e) > 0 and idx_1e[0] > 0:
            x_c_1e.append(X[idx_1e[0]])
            y_c_1e.append(y_norm[idx_1e[0]])
            
        # 2. 10% threshold (0.1)
        idx_01 = np.where(y_norm <= 0.1)[0]
        if len(idx_01) > 0 and idx_01[0] > 0:
            x_c_01.append(X[idx_01[0]])
            y_c_01.append(y_norm[idx_01[0]])
            
        # 3. Local slope thresholds (using fast numpy diff and convolve)
        valid = (X > 0) & (y_norm > 1e-12)
        X_v = X[valid]
        y_v = y_norm[valid]
        
        if len(X_v) > 20:
            ln_x = np.log(X_v)
            ln_y = np.log(y_v)
            
            # Simple difference slope
            dy = np.diff(ln_y)
            dx = np.diff(ln_x)
            slopes_raw = dy / dx
            
            # Smooth slopes with a convolve window
            w = 31
            slopes = np.convolve(slopes_raw, np.ones(w)/w, mode='same')
            
            # Pad to match length of X_v
            slopes = np.append(slopes, slopes[-1])
            
            # slope = -0.5
            idx_s05 = np.where(slopes <= -0.5)[0]
            if len(idx_s05) > 0:
                x_c_slope_05.append(X_v[idx_s05[0]])
                y_c_slope_05.append(y_norm[X_v[idx_s05[0]]])
                
            # slope = -1.0
            idx_s1 = np.where(slopes <= -1.0)[0]
            if len(idx_s1) > 0:
                x_c_slope_1.append(X_v[idx_s1[0]])
                y_c_slope_1.append(y_norm[X_v[idx_s1[0]]])
            
    plt.plot(x_c_1e, y_c_1e, 'ro-', label='Threshold 1/e (approx 0.368)')
    plt.plot(x_c_01, y_c_01, 'go-', label='Threshold 10%')
    plt.plot(x_c_slope_05, y_c_slope_05, 'bo-', label='Slope threshold -0.5')
    plt.plot(x_c_slope_1, y_c_slope_1, 'mo-', label='Slope threshold -1.0')
    
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('Distance x')
    plt.ylabel('Normalized Amplitude')
    plt.title('Comparison of Crossover Detection Methods (Fast)')
    plt.legend()
    plt.xlim(1, 3000)
    plt.ylim(1e-4, 2.0)
    
    plt.savefig('crossover_test.png', dpi=300)
    print("Plot saved as crossover_test.png")

if __name__ == "__main__":
    main()
