import numpy as np
import os

def main():
    base_dir = "/Users/emericmellet/Desktop/Emeric/School/UPS/M1 SGM/Stage/ENSTA/Rapport/code"
    
    file1 = os.path.join(base_dir, "parametric_sweep", "data", "envelopes_eps=1.4_N=6000.csv")
    file2 = os.path.join(base_dir, "parametric_sweep", "data", "envelopes_eps=1.4_N=6000_test.csv")
    
    for f in [file1, file2]:
        if os.path.exists(f):
            data = np.loadtxt(f, delimiter=',', skiprows=1)
            print(f"File: {os.path.basename(f)}")
            print(f"  Shape: {data.shape}")
            print(f"  File size on disk: {os.path.getsize(f)} bytes")
        else:
            print(f"File not found: {f}")

if __name__ == "__main__":
    main()
