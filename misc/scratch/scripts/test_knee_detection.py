import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

def get_data_file(eps):
    return f"code/parametric_sweep/data/envelopes_eps={eps}_N=6000.csv"

eps = 1.4
N = 6000
data_file = get_data_file(eps)
data = np.loadtxt(data_file, delimiter=',', skiprows=1)
omegas = data[:, 0]
envelopes = data[:, 1 + N//2 : 1 + N]
X = np.arange(0, N//2)

idx = 10 
y = envelopes[idx, :]
omega = omegas[idx]

trunc = 2500
X_t = X[1:trunc]
y_t = y[1:trunc]

u = np.log(X_t)
v = np.log(y_t)

v_smooth = savgol_filter(v, window_length=51, polyorder=3)

du = np.gradient(u)
dv = np.gradient(v_smooth, u)
d2v = np.gradient(dv, u)

curvature = np.abs(d2v) / (1 + dv**2)**(1.5)

peak_idx = np.argmax(curvature)
x_c = X_t[peak_idx]
y_c = y_t[peak_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.loglog(X_t, y_t, label="Raw Data")
ax1.loglog(X_t, np.exp(v_smooth), label="Smoothed")
ax1.loglog(x_c, y_c, 'ro', markersize=8, label="Detected Knee")
ax1.set_title(f"Envelope at omega={omega:.3f}")
ax1.grid(True)
ax1.legend()

ax2.plot(X_t, curvature, label="Curvature in Log-Log")
ax2.plot(x_c, curvature[peak_idx], 'ro')
ax2.set_xscale('log')
ax2.set_title("Curvature Peak")
ax2.grid(True)

plt.savefig("code/scratch/plots_and_animations/knee_detection_test.png")
