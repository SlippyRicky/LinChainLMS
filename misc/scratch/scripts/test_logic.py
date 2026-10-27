import numpy as np

N = 6000
k0, m0 = 1.0, 1.0
q_modes = np.arange(1, N, 2)
OMEGAS = 2.0 * np.sqrt(k0 / m0) * np.sin(q_modes * np.pi / (2.0 * (N + 1)))
OMEGAS = OMEGAS[::150]

omegas_pure = 2.0 * np.sqrt(k0 / m0) * np.sin(np.arange(1, N+1) * np.pi / (2.0 * (N + 1)))
indices = [np.argmin(np.abs(omegas_pure - o)) for o in OMEGAS]

print(indices)
print(q_modes[::150] - 1)
