"""
chain.py — Moteur physique orienté objet pour chaînes harmoniques 1D désordonnées.

Modèle numérique vérifié : intégration temporelle de Verlet à vitesse (Velocity-Verlet)
avec conditions aux limites absorbantes (ABC / PML quadratique).

Ce moteur unifié supporte :
    - Désordre de raideur ('stiff'), de masse ('mass'), ou combiné ('combined')
    - Excitation de type paquet d'onde ('wavepacket') ou impact ('impact')
    - Extraction de la longueur de localisation par ajustement exponentiel de l'enveloppe

Référence : sim_paquet_comparaison_Env+Etot.py (modèle vérifié)
"""
import numpy as np
from scipy.optimize import curve_fit


class DisorderedChain:
    """
    Chaîne harmonique 1D désordonnée avec intégration de Verlet.
    
    Parameters
    ----------
    N : int
        Nombre de masses dans la chaîne.
    epsilon : float
        Amplitude maximale du désordre (distribution uniforme [-ε/2, +ε/2]).
    disorder_type : str
        Type de désordre : 'stiff', 'mass', ou 'combined'.
    seed : int
        Germe du générateur de nombres aléatoires pour la reproductibilité.
    m0 : float
        Masse nominale (unités réduites, défaut = 1.0).
    k0 : float
        Raideur nominale (unités réduites, défaut = 1.0).
    """
    
    def __init__(self, N=1500, epsilon=0.5, disorder_type='stiff', seed=42,
                 m0=1.0, k0=1.0):
        self.N = N
        self.epsilon = epsilon
        self.disorder_type = disorder_type
        self.seed = seed
        self.m0 = m0
        self.k0 = k0
        
        # Derived physical constants
        self.omega_max = 2.0 * np.sqrt(k0 / m0)
        
        # Generate disordered chain
        self._build_chain()
    
    def _build_chain(self):
        """Generate the disordered mass and stiffness arrays."""
        rng = np.random.RandomState(self.seed)
        
        # Masses (N elements)
        self.masses = np.ones(self.N) * self.m0
        # Stiffnesses (N+1 bonds: wall—m0—m1—...—m_{N-1}—wall)
        self.k_bonds = np.ones(self.N + 1) * self.k0
        
        if self.disorder_type in ('mass', 'combined'):
            self.masses *= (1.0 + self.epsilon * (rng.rand(self.N) - 0.5))
        
        if self.disorder_type in ('stiff', 'combined'):
            self.k_bonds *= (1.0 + self.epsilon * (rng.rand(self.N + 1) - 0.5))
    
    def simulate(self, excitation='wavepacket', omega_ratio=0.25,
                 dt=0.05, steps=40000, downsample=200,
                 t_pulse=80.0, l_abc=100, gamma_max=0.1,
                 excitation_node=None):
        """
        Run a full time-domain Velocity-Verlet simulation.
        
        Parameters
        ----------
        excitation : str
            'wavepacket' (Gaussian-modulated sinusoidal burst) or 'impact' (Dirac-like impulse).
        omega_ratio : float
            Excitation frequency as fraction of omega_max (used for wavepacket).
        dt : float
            Time step (reduced units).
        steps : int
            Total number of integration steps.
        downsample : int
            Snapshot frequency for history recording.
        t_pulse : float
            Gaussian envelope duration (wavepacket mode).
        l_abc : int
            Thickness of the absorbing boundary layer (sites).
        gamma_max : float
            Maximum damping coefficient in the ABC layer.
        excitation_node : int or None
            Site index for excitation. If None, defaults to l_abc + 50.
        
        Returns
        -------
        result : dict
            Dictionary containing:
            - 'u_hist': (n_frames, N) displacement snapshots
            - 'e_hist': (n_frames,) total energy time series
            - 'envelope': (N,) maximum displacement envelope
            - 'times': (n_frames,) time values
            - 'excitation_node': int, site where excitation was applied
        """
        N = self.N
        m = self.masses
        k = self.k_bonds
        omega_ext = omega_ratio * self.omega_max
        
        if excitation_node is None:
            excitation_node = l_abc + 50
        
        # Absorbing boundary layer (quadratic profile)
        g_profile = np.zeros(N)
        for i in range(min(l_abc, N // 2)):
            g_profile[i] = gamma_max * ((l_abc - i) / l_abc) ** 2
            g_profile[N - 1 - i] = gamma_max * ((l_abc - i) / l_abc) ** 2
        
        # Allocate state
        u = np.zeros(N)
        v = np.zeros(N)
        
        n_frames = steps // downsample
        u_hist = np.zeros((n_frames, N))
        e_hist = np.zeros(n_frames)
        envelope = np.zeros(N)
        
        def compute_acceleration(u_c, v_c, t_c):
            """Compute accelerations from forces (springs + ABC + excitation)."""
            f = np.zeros(N)
            # Internal spring forces (vectorized)
            # Convention: k[i] is the bond between mass i-1 and mass i
            # Mass 0: left wall bond k[0], right bond k[1]
            f[0] = -k[0] * u_c[0] - k[1] * (u_c[0] - u_c[1])
            # Masses 1..N-2 (vectorized): left bond k[1:N-1], right bond k[2:N]
            f[1:N-1] = (-k[1:N-1] * (u_c[1:N-1] - u_c[0:N-2])
                        - k[2:N]   * (u_c[1:N-1] - u_c[2:N]))
            # Mass N-1: left bond k[N-1], right wall bond k[N]
            f[N-1] = -k[N-1] * (u_c[N-1] - u_c[N-2]) - k[N] * u_c[N-1]
            
            # Absorbing boundary damping
            f -= g_profile * v_c
            
            # External excitation
            if excitation == 'wavepacket':
                if t_c < 2 * t_pulse:
                    amp = np.exp(-(t_c - t_pulse) ** 2 / (2 * (t_pulse / 6) ** 2))
                    f[excitation_node] += 1.0 * amp * np.sin(omega_ext * t_c)
            elif excitation == 'impact':
                # Short Gaussian impulse (broadband)
                t_impact = 5.0 * dt  # Very short impulse centered near t=0
                sigma_impact = 2.0 * dt
                if t_c < 10 * dt:
                    f[excitation_node] += 5.0 * np.exp(-(t_c - t_impact) ** 2 / (2 * sigma_impact ** 2))
            
            return f / m
        
        # Velocity-Verlet integration loop
        a = compute_acceleration(u, v, 0.0)
        for step in range(steps):
            t = step * dt
            v_half = v + 0.5 * dt * a
            u += v_half * dt
            
            # Update envelope (max displacement ever reached at each site)
            np.maximum(envelope, np.abs(u), out=envelope)
            
            a = compute_acceleration(u, v_half, t + dt)
            v = v_half + 0.5 * dt * a
            
            # Record snapshot
            if step % downsample == 0:
                idx = step // downsample
                if idx < n_frames:
                    u_hist[idx] = u.copy()
                    ek = 0.5 * np.sum(m * v ** 2)
                    ep = (0.5 * k[0] * u[0] ** 2
                          + 0.5 * np.sum(k[1:N] * (u[1:] - u[:-1]) ** 2)
                          + 0.5 * k[N] * u[-1] ** 2)
                    e_hist[idx] = ek + ep
        
        times = np.arange(n_frames) * downsample * dt
        
        return {
            'u_hist': u_hist,
            'e_hist': e_hist,
            'envelope': envelope,
            'times': times,
            'excitation_node': excitation_node,
        }
    
    def extract_localization_length(self, envelope, excitation_node, fit_range=None):
        """
        Extract the localization length λ = 1/γ by fitting an exponential
        decay to the displacement envelope on both sides of the excitation site.
        
        Parameters
        ----------
        envelope : np.ndarray
            Maximum displacement envelope (N,).
        excitation_node : int
            Index of the excitation site.
        fit_range : tuple or None
            (min_distance, max_distance) from excitation site for fitting.
            If None, uses (20, N//3).
        
        Returns
        -------
        loc_length : float
            Estimated localization length (in number of sites).
        gamma : float
            Estimated Lyapunov exponent (inverse localization length).
        fit_quality : float
            R² coefficient of the exponential fit.
        """
        if fit_range is None:
            fit_range = (20, self.N // 3)
        
        # Work on the right side of the excitation
        env_right = envelope[excitation_node:]
        if len(env_right) < fit_range[1]:
            fit_range = (fit_range[0], len(env_right) - 5)
        
        distances = np.arange(len(env_right))
        
        # Select fitting window
        mask = (distances >= fit_range[0]) & (distances <= fit_range[1])
        d_fit = distances[mask]
        e_fit = env_right[mask]
        
        # Avoid log(0)
        e_fit = np.maximum(e_fit, 1e-30)
        
        # Linear fit in log space: log(env) = log(A) - gamma * d
        try:
            log_env = np.log(e_fit)
            coeffs = np.polyfit(d_fit, log_env, 1)
            gamma = -coeffs[0]  # Slope is -gamma
            log_A = coeffs[1]
            
            # Compute R²
            log_pred = np.polyval(coeffs, d_fit)
            ss_res = np.sum((log_env - log_pred) ** 2)
            ss_tot = np.sum((log_env - np.mean(log_env)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            
            loc_length = 1.0 / gamma if gamma > 0 else np.inf
            
            return loc_length, gamma, r_squared
        except Exception:
            return np.inf, 0.0, 0.0
    
    def extract_penetration_length(self, envelope, excitation_node):
        """
        Extract the penetration length L_p from the envelope.
        L_p is the distance from the excitation node at which the envelope
        amplitude drops to 1/e (i.e. approx 36.79%) of its value at the excitation node.
        
        Using linear interpolation for sub-site precision.
        """
        u_source = envelope[excitation_node]
        if u_source <= 0:
            return 0.0
            
        target = u_source / np.exp(1.0)
        
        # We look to the right of the excitation node
        env_right = envelope[excitation_node:]
        distances = np.arange(len(env_right))
        
        # Find first index where envelope is below or equal to target
        indices = np.where(env_right <= target)[0]
        if len(indices) == 0:
            return float(len(env_right))
            
        first_idx = indices[0]
        
        if first_idx == 0:
            return 0.0
            
        # Linear interpolation
        y1 = env_right[first_idx - 1]
        y2 = env_right[first_idx]
        x1 = distances[first_idx - 1]
        x2 = distances[first_idx]
        
        if y1 != y2:
            x_target = x1 + (target - y1) * (x2 - x1) / (y2 - y1)
        else:
            x_target = x1
            
        return x_target

    def __repr__(self):
        return (f"DisorderedChain(N={self.N}, ε={self.epsilon}, "
                f"type='{self.disorder_type}', seed={self.seed})")
