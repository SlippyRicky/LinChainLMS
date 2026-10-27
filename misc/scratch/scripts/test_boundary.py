import numpy as np
import os

data_path = "/Users/emericmellet/Desktop/Emeric/School/UPS/M1 SGM/Stage/ENSTA/Rapport/code/figures_generation/data/wp_vs_impact_comparison.npz"
data = np.load(data_path)

for key in ["impact", "wp_low", "wp_high"]:
    arr = data[key]
    print(f"\n--- {key} ---")
    print(f"Shape: {arr.shape}")
    print("At t = 0, first 10 columns:")
    print(arr[0, :10])
    print("At t = 0, last 10 columns:")
    print(arr[0, -10:])
    print("Maximum at n=0 for t in [0, 50] (first 50 steps):")
    print(np.max(arr[:50, 0]))
    print("Maximum in whole array:")
    print(np.max(arr))
