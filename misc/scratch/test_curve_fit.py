import numpy as np
from scipy.optimize import curve_fit
import os

def find_data_file(eps, N, test=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    suffix = "_test" if test else ""
    filename = f"envelopes_eps={eps}_N={N}{suffix}.csv"
    paths_to_check = [
        os.path.join(project_root, "code", "parametric_sweep", "data", filename),
        os.path.join(project_root, "code", "scratch", filename),
        os.path.join(project_root, "code", "scratch", "Analyse", "data", filename),
        os.path.join(script_dir, filename),
        os.path.join(script_dir, "Analyse", "data", filename)
    ]
    for path in paths_to_check:
        if os.path.exists(path):
            return path
    return None

disorder_levels = [0.1, 0.7, 1.4]
N = 6000

def exp_func(x, A, B):
    return A * np.exp(B * x)

for eps in disorder_levels:
    data_file = find_data_file(eps, N, test=False)
    if not data_file:
        data_file = find_data_file(eps, N, test=True)
        
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
    valid = ~np.isnan(x_c_vals) & (x_c_vals > 0)
    
    # Filtering artifacts:
    # 1. Upper boundary saturation (x_c > 2800)
    # 2. Lower boundary floor artifacts (x_c < 10 or 15)
    valid = valid & (x_c_vals < 2800) & (x_c_vals > 10)
    
    omega_v = omegas[valid]
    x_c_v = x_c_vals[valid]
    
    if len(omega_v) > 2:
        from scipy.stats import linregress
        slope, intercept, _, _, _ = linregress(omega_v, np.log(x_c_v))
        p0 = (np.exp(intercept), slope)
        
        try:
            popt, pcov = curve_fit(exp_func, omega_v, x_c_v, p0=p0)
            A, B = popt
            
            residuals = x_c_v - exp_func(omega_v, A, B)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((x_c_v - np.mean(x_c_v))**2)
            r2 = 1 - (ss_res / ss_tot)
            
            print(f"[{eps}] Curve Fit: x_c = {A:.1f} * exp({B:.3f} * omega) (R2 = {r2:.4f}), Points used: {len(omega_v)}")
        except Exception as e:
            print(f"[{eps}] Fit failed: {e}")
