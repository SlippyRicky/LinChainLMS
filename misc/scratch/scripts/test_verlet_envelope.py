import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

def test_verlet():
    N = 300
    epsilon_max = 0.3
    M0, K0 = 1.0, 1.0
    omega = np.sqrt(2)
    g = 0.01  # small physical damping
    
    np.random.seed(42)
    # Mass disorder
    epsilon = np.random.uniform(-epsilon_max, epsilon_max, N)
    m = M0 * (1 + epsilon)
    K = np.ones(N + 1) * K0
    
    # Velocity Verlet Simulation
    u = np.zeros(N)
    v = np.zeros(N)
    a = np.zeros(N)
    envelope = np.zeros(N)
    
    dt = 0.05
    STEPS = 60000
    
    print("Running time-domain Velocity Verlet simulation...")
    for step in range(STEPS):
        t = step * dt
        
        # 1. Update displacements (excluding site 0 which is prescribed)
        for n in range(1, N):
            u[n] += v[n] * dt + 0.5 * a[n] * dt**2
            
        # Prescribe boundary condition at site 0
        u[0] = np.sin(omega * t)
        
        # Save old accelerations
        a_old = a.copy()
        
        # 2. Compute new accelerations
        for n in range(1, N - 1):
            force = K[n] * (u[n+1] - u[n]) - K[n-1] * (u[n] - u[n-1]) - g * v[n]
            a[n] = force / m[n]
            
        # Last site N-1: clamped
        force_last = K[N-1] * (0.0 - u[N-1]) - K[N-2] * (u[N-1] - u[N-2]) - g * v[N-1]
        a[N-1] = force_last / m[N-1]
        
        # 3. Update velocities
        for n in range(1, N):
            v[n] += 0.5 * dt * (a_old[n] + a[n])
            
        # Record the maximum displacement envelope during the steady state (last 15000 steps)
        if step > STEPS - 15000:
            for n in range(N):
                abs_u = abs(u[n])
                if abs_u > envelope[n]:
                    envelope[n] = abs_u

    nodes = np.arange(N)
    # Fit log(envelope) to get Lyapunov exponent
    # Let's fit from n = 10 to n = 150 to avoid edge effects and noise
    fit_mask = (nodes >= 10) & (nodes <= 150) & (envelope > 1e-5)
    fit_nodes = nodes[fit_mask]
    fit_u = envelope[fit_mask]
    
    slope, intercept, r_value, p_value, std_err = linregress(fit_nodes, np.log(fit_u))
    gamma_fit = -slope
    
    print(f"Fitted Lyapunov exponent gamma = {gamma_fit:.5f}")
    print(f"R-squared of the fit = {r_value**2:.5f}")
    
    # Check what the theoretical Lyapunov exponent is (Matsuda-Ishii)
    # gamma = (epsilon^2 / 96) * omega^2
    gamma_theory = (epsilon_max**2 / 96.0) * (omega**2)
    print(f"Theoretical Matsuda-Ishii exponent gamma_theory = {gamma_theory:.5f}")

if __name__ == "__main__":
    test_verlet()
