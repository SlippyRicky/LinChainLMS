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
    
    plt.figure(figsize=(10, 8))
    
    # Plot all envelopes in log-log
    for i in range(len(omegas)):
        y = envelopes[i, :]
        y_norm = y / y[0]
        plt.loglog(X[1:], y_norm[1:], color='gray', alpha=0.3)
        
    # We will find the crossover point for each frequency using different thresholds
    x_c_1e = []
    y_c_1e = []
    
    x_c_50 = []
    y_c_50 = []
    
    x_c_slope_05 = []
    y_c_slope_05 = []
    
    x_c_slope_02 = []
    y_c_slope_02 = []
    
    for i in range(len(omegas)):
        y = envelopes[i, :]
        y_norm = y / y[0]
        
        # 1. 1/e threshold
        idx_1e = np.where(y_norm <= 1.0 / np.e)[0]
        if len(idx_1e) > 0 and idx_1e[0] > 0:
            x_c_1e.append(X[idx_1e[0]])
            y_c_1e.append(y_norm[idx_1e[0]])
            
        # 2. 50% threshold
        idx_50 = np.where(y_norm <= 0.5)[0]
        if len(idx_50) > 0 and idx_50[0] > 0:
            x_c_50.append(X[idx_50[0]])
            y_c_50.append(y_norm[idx_50[0]])
            
        # 3. Local slope thresholds
        valid = (X > 0) & (y_norm > 0)
        X_v = X[valid]
        y_v = y_norm[valid]
        ln_x = np.log(X_v)
        ln_y = np.log(y_v)
        
        slopes = []
        window = 15
        for k in range(len(ln_x)):
            i_min = max(0, k - window)
            i_max = min(len(ln_x), k + window)
            if i_max - i_min > 2:
                slope, _, _, _, _ = linregress(ln_x[i_min:i_max], ln_y[i_min:i_max])
                slopes.append(slope)
            else:
                slopes.append(0.0)
        slopes = np.array(slopes)
        
        # slope = -0.2
        idx_s02 = np.where(slopes <= -0.2)[0]
        if len(idx_s02) > 0:
            x_c_slope_02.append(X_v[idx_s02[0]])
            y_c_slope_02.append(y_norm[X_v[idx_s02[0]]])
            
        # slope = -0.5
        idx_s05 = np.where(slopes <= -0.5)[0]
        if len(idx_s05) > 0:
            x_c_slope_05.append(X_v[idx_s05[0]])
            y_c_slope_05.append(y_norm[X_v[idx_s05[0]]])
            
    plt.plot(x_c_1e, y_c_1e, 'ro-', label='Threshold 1/e (approx 0.368)')
    plt.plot(x_c_50, y_c_50, 'go-', label='Threshold 50%')
    plt.plot(x_c_slope_02, y_c_slope_02, 'bo-', label='Slope threshold -0.2')
    plt.plot(x_c_slope_05, y_c_slope_05, 'mo-', label='Slope threshold -0.5')
    
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('Distance x')
    plt.ylabel('Normalized Amplitude')
    plt.title('Comparison of Crossover Detection Methods')
    plt.legend()
    plt.xlim(1, 3000)
    plt.ylim(1e-4, 2.0)
    
    plt.savefig('crossover_test.png', dpi=300)
    print("Plot saved as crossover_test.png")

if __name__ == "__main__":
    main()
