import numpy as np

def run_sim(N):
    T_max = 1000
    dt = 0.05
    steps = int(T_max / dt)
    
    np.random.seed(42)
    epsilon = 0.5
    m0 = 1.0
    m_n = m0 * (1.0 + epsilon * (np.random.rand(N) - 0.5))
    K = 1.0
    k_bonds = np.ones(N + 1) * K
    
    center = N // 2
    sigma = 10.0
    q = 0.15 * np.pi
    
    nodes = np.arange(N)
    u0 = np.exp(-(nodes - center)**2 / (2 * sigma**2))
    u0 *= np.cos(q * (nodes - center))
    u0 = u0 / np.linalg.norm(u0)
    
    u = u0.copy()
    v = np.zeros(N)
    
    # PML profile
    l_abc = 100
    gamma_max = 0.20
    g_profile = np.zeros(N)
    for i in range(l_abc):
        g_profile[i] = gamma_max * ((l_abc - i) / l_abc)**2
        g_profile[N-1-i] = gamma_max * ((l_abc - i) / l_abc)**2
        
    def get_acc(u_c, v_c):
        f = np.zeros(N)
        f[0] = -k_bonds[0]*u_c[0] - k_bonds[1]*(u_c[0]-u_c[1])
        f[1:N-1] = -k_bonds[1:N-1]*(u_c[1:N-1]-u_c[0:N-2]) - k_bonds[2:N]*(u_c[1:N-1]-u_c[2:N])
        f[N-1] = -k_bonds[N-1]*(u_c[N-1]-u_c[N-2]) - k_bonds[N]*u_c[N-1]
        f -= g_profile * v_c
        return f / m_n
        
    a = get_acc(u, v)
    
    for step in range(steps):
        v_half = v + 0.5 * dt * a
        u += v_half * dt
        a = get_acc(u, v_half)
        v = v_half + 0.5 * dt * a
        
    # Calculate energy variance (spatial spread)
    energy = u**2
    energy /= np.sum(energy)
    mean = np.sum(nodes * energy)
    var = np.sum(((nodes - mean)**2) * energy)
    std = np.sqrt(var)
    
    # Also find where energy drops below a threshold relative to the peak
    peak_idx = np.argmax(energy)
    peak_val = energy[peak_idx]
    threshold = 0.01 * peak_val
    left_bound = np.where(energy[:peak_idx] > threshold)[0]
    right_bound = np.where(energy[peak_idx:] > threshold)[0] + peak_idx
    
    width_nodes = (right_bound[-1] if len(right_bound) > 0 else N) - (left_bound[0] if len(left_bound) > 0 else 0)
    
    print(f"N = {N}: Center = {center}")
    print(f"  Physical Mean Position: {mean:.2f}")
    print(f"  Physical Std Dev of energy: {std:.2f} nodes")
    print(f"  Width (1% of peak threshold): {width_nodes} nodes")
    print(f"  Relative Std Dev (Std / N): {std/N:.4f}")
    print(f"  Relative Width (Width / N): {width_nodes/N:.4f}")

run_sim(6000)
run_sim(8000)
