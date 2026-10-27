import numpy as np
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

for eps in disorder_levels:
    data_file = find_data_file(eps, N, test=False)
    if not data_file:
        data_file = find_data_file(eps, N, test=True)
        
    data = np.loadtxt(data_file, delimiter=',', skiprows=1)
    omegas = data[:, 0]
    envelopes = data[:, 1 + N//2 : 1 + N]
    X = np.arange(0, N//2)
    
    x_c_vals = []
    y_c_vals = []
    y_source_vals = []
    
    for i in range(len(omegas)):
        y = envelopes[i, :]
        y_source = y[0]
        y_source_vals.append(y_source)
        
        # Knee 1/e
        threshold = y_source / np.e
        idx_c = np.where(y <= threshold)[0]
        if len(idx_c) > 0 and idx_c[0] > 0:
            x_c_vals.append(X[idx_c[0]])
            y_c_vals.append(y[idx_c[0]])
        else:
            x_c_vals.append(np.nan)
            y_c_vals.append(np.nan)
            
    print(f"\n--- EPS = {eps} ---")
    print("Omegas:", np.array(omegas[:10]).round(3))
    print("x_c_vals:", x_c_vals[:10])
    print("y_c_vals (knee amplitude):", np.array(y_c_vals[:10]).round(4))
    print("y_source_vals:", np.array(y_source_vals[:10]).round(4))
