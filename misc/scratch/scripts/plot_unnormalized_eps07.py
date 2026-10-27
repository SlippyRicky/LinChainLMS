import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    base_dir = "/Users/emericmellet/Desktop/Emeric/School/UPS/M1 SGM/Stage/ENSTA/Rapport/code"
    data_file = os.path.join(base_dir, "parametric_sweep", "data", "envelopes_eps=0.7_N=6000.csv")
    
    if not os.path.exists(data_file):
        print("Data file not found.")
        return
        
    data = np.loadtxt(data_file, delimiter=',', skiprows=1)
    omegas = data[:, 0]
    envelopes = data[:, 3001:6001]
    X = np.arange(0, 3000)
    
    plt.figure(figsize=(8, 6))
    
    # Plot all envelopes in log-log
    for i in range(len(omegas)):
        y = envelopes[i, :]
        plt.loglog(X[1:], y[1:], linewidth=0.8)
        
    plt.grid(True, which='both', alpha=0.3)
    plt.xlabel('Distance x')
    plt.ylabel('Amplitude Y')
    plt.title('Unnormalized Log-Log Envelopes (eps=0.7) in Python')
    plt.xlim(1, 3000)
    plt.ylim(1e-4, 100.0)
    
    plt.savefig('unnormalized_eps07_test.png', dpi=300)
    print("Plot saved as unnormalized_eps07_test.png")

if __name__ == "__main__":
    main()
