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
    
    print("Frequencies:")
    print(omegas)
    
    print("\nFirst 10 columns of envelopes for the first 3 frequencies:")
    for i in range(3):
        print(f"Freq = {omegas[i]:.4f}:")
        print(envelopes[i, :10])

if __name__ == "__main__":
    main()
