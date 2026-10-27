import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_local_slope(x, y, window=50):
    """Calcule la pente locale d(ln y) / d(ln x) par lissage."""
    # On évite log(0)
    valid = (x > 0) & (y > 0)
    x_v = x[valid]
    y_v = y[valid]
    
    ln_x = np.log(x_v)
    ln_y = np.log(y_v)
    
    slopes = np.zeros_like(ln_x)
    for i in range(len(ln_x)):
        i_min = max(0, i - window)
        i_max = min(len(ln_x), i + window)
        
        if i_max - i_min > 2:
            p = np.polyfit(ln_x[i_min:i_max], ln_y[i_min:i_max], 1)
            slopes[i] = p[0]
            
    return x_v, slopes

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_file = os.path.join(base_dir, 'Analyse', 'data', 'envelopes_eps=1.4_N=6000.csv')
    
    if not os.path.exists(data_file):
        test_file = os.path.join(base_dir, 'Analyse', 'data', 'envelopes_eps=1.4_N=6000_test.csv')
        if os.path.exists(test_file):
            data_file = test_file
        else:
            print(f"Fichier de données introuvable.")
            return

    data = np.loadtxt(data_file, delimiter=',', skiprows=1)
    N = 6000
    X = np.arange(N//2, N) - N//2
    
    # Prendre la moyenne des enveloppes ou une enveloppe spécifique
    # On saute la première colonne
    envelopes = data[:, 1 + N//2 : 1 + N]
    
    # Prendre une fréquence médiane
    mid_idx = envelopes.shape[0] // 2
    Y = envelopes[mid_idx, :]
    
    # Normalisation
    Y = Y / Y[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Plot Log-Log slope
    x_v, slopes = calculate_local_slope(X, Y, window=20)
    ax = axes[0]
    ax.plot(x_v, slopes, label="Local slope")
    ax.axhline(-0.5, color='r', linestyle='--', label="Half inverse power (-0.5)")
    # Exponential slope would be -x/xi, which is a straight line downwards in linear x
    # Here x axis is log or linear. Let's use semilogx for x
    ax.set_xscale('log')
    ax.set_xlabel('Distance x (log)')
    ax.set_ylabel('Slope d(ln y) / d(ln x)')
    ax.set_title('Local Log-Log Slope')
    ax.grid(True, alpha=0.5)
    ax.legend()
    
    # 2. Correlation log(-log(y)) vs log(x)
    # Check if stretched exponential
    ax = axes[1]
    valid2 = (X > 0) & (Y > 0) & (Y < 1)
    x_v2 = X[valid2]
    y_v2 = Y[valid2]
    ln_x2 = np.log(x_v2)
    ln_ln_y = np.log(-np.log(y_v2))
    
    ax.plot(ln_x2, ln_ln_y, '.')
    ax.set_xlabel('log(x)')
    ax.set_ylabel('log(-log(y))')
    ax.set_title('Test: Stretched Exponential (KWW)\nStraight line means stretched exp')
    ax.grid(True, alpha=0.5)
    
    # 3. Participation Ratio & Characteristic lengths
    # PR = (sum u^2)^2 / sum u^4
    PR = (np.sum(Y**2))**2 / np.sum(Y**4)
    
    # Exp fit length xi
    valid_exp = X[1:1000] # Fit near source
    y_exp = Y[1:1000]
    p = np.polyfit(valid_exp, np.log(y_exp), 1)
    xi_exp = -1.0 / p[0]
    
    ax = axes[2]
    ax.text(0.1, 0.8, f"Longueur de localisation:", fontsize=12)
    ax.text(0.1, 0.6, f"PR (Participation Ratio) = {PR:.1f} sites", fontsize=12)
    ax.text(0.1, 0.4, f"Xi (Log-Lin decay near src) = {xi_exp:.1f} sites", fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    output_png = os.path.join(base_dir, 'scratch', 'analyze_slopes.png')
    plt.savefig(output_png, dpi=300)
    print(f"Analyse des pentes générée : {output_png}")

if __name__ == "__main__":
    main()
