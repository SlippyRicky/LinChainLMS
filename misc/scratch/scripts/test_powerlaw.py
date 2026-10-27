import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit

disorder_levels = [0.1, 0.7, 1.4]
N = 6000

def get_data_file(eps):
    # Enforce using the actual complete data file
    return f"code/parametric_sweep/data/envelopes_eps={eps}_N=6000.csv"

for eps in disorder_levels:
    data_file = get_data_file(eps)
    data = np.loadtxt(data_file, delimiter=',', skiprows=1)
    omegas = data[:, 0]
    envelopes = data[:, 1 + N//2 : 1 + N]
    X = np.arange(0, N//2)
    
    x_c_vals = []
    for i in range(len(omegas)):
        y = envelopes[i, :]
        threshold = y[0] / np.e
        idx_c = np.where(y <= threshold)[0]
        if len(idx_c) > 0 and idx_c[0] > 0:
            x_c_vals.append(X[idx_c[0]])
        else:
            x_c_vals.append(np.nan)
            
    x_c_vals = np.array(x_c_vals)
    valid = ~np.isnan(x_c_vals) & (x_c_vals > 10) & (x_c_vals < 2800)
    
    omega_v = omegas[valid]
    x_c_v = x_c_vals[valid]
    
    if len(omega_v) > 2:
        # Power-Law Fit in log-log space: ln(x_c) = -B * ln(omega) + ln(A)
        slope, intercept, r_val_log, _, _ = linregress(np.log(omega_v), np.log(x_c_v))
        A_pl = np.exp(intercept)
        B_pl = -slope
        
        # Calculate linear R2 for power-law
        res_pl = x_c_v - (A_pl * omega_v**(-B_pl))
        ss_res_pl = np.sum(res_pl**2)
        ss_tot = np.sum((x_c_v - np.mean(x_c_v))**2)
        r2_pl_linear = 1 - (ss_res_pl / ss_tot)
        
        print(f"[{eps}] POWER-LAW: x_c = {A_pl:.1f} * omega^(-{B_pl:.3f})")
        print(f"      Log-Log R2 = {r_val_log**2:.4f}, Linear R2 = {r2_pl_linear:.4f}")
