import numpy as np
import os

def main():
    base_dir = "/Users/emericmellet/Desktop/Emeric/School/UPS/M1 SGM/Stage/ENSTA/Rapport/code"
    data_file = os.path.join(base_dir, "parametric_sweep", "data", "envelopes_eps=1.4_N=6000.csv")
    
    if not os.path.exists(data_file):
        print("Data file not found.")
        return
        
    data = np.loadtxt(data_file, delimiter=',', skiprows=1)
    omegas = data[:, 0]
    envelopes = data[:, 3001:6001]
    
    for i in range(len(omegas)):
        y = envelopes[i, :]
        y_norm = y / y[0]
        print(f"Freq = {omegas[i]:.4f}: y(0) = {y[0]:.4f} | y(10) = {y[10]:.4f} (norm={y_norm[10]:.4f}) | y(100) = {y[100]:.4f} (norm={y_norm[100]:.4f}) | y(1000) = {y[1000]:.4f} (norm={y_norm[1000]:.4f})")

if __name__ == "__main__":
    main()
