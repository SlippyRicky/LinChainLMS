import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import os
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# --- STYLE CONFIGURATION (Premium Manuscript) ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 21,
    "legend.fontsize": 15,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.2
})

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
import plot_config
fig_dir = os.path.join(plot_config.FIGURES_ROOT, '02_linear_disorder')
# Directory creation handled by plot_config
data_dir = plot_config.DATA_DIR


def plot_spectral_broadening():
    """Figure 13: Spectral Function and 1D Slices showing frequency detuning"""
    data_path = os.path.join(data_dir, "disordered_modes_N500_eps0.4.npz")
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return
    data = np.load(data_path)
    omega = data['omega']
    eigvecs = data['eigvecs']
    
    fft_vals = np.fft.rfft(eigvecs, axis=0)
    psd = np.abs(fft_vals)**2
    psd_norm = psd / np.max(psd, axis=0, keepdims=True)
    
    N_q = psd.shape[0]
    q_norm = np.linspace(0, 1, N_q)
    
    # Target q values for cuts
    q_targets = [0.2, 0.4, 0.6, 0.8]
    indices = [np.argmin(np.abs(q_norm - qt)) for qt in q_targets]
    slice_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"] # Distinct colors for slices
    
    # Create a 2-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.76, 4.94))
    
    # --- PANEL 1: 2D Spectral Function Heatmap ---
    Q, W = np.meshgrid(q_norm, omega)
    heatmap = ax1.pcolormesh(Q, W, psd_norm.T, cmap='afmhot_r', shading='nearest', vmin=0, vmax=1)
    
    # Ballistic dispersion curve
    # We plot it as a clean black dashed line, which stands out perfectly on the light inferno_r background
    w_theoretical = 2.0 * np.sin(q_norm * np.pi / 2)
    ax1.plot(q_norm, w_theoretical, color='black', linestyle=':', linewidth=2, alpha=0.8,
             label=r'Dispersion balistique ($\sigma=0$)')
    
    # Overlay lines of cuts
    for qt, color in zip(q_targets, slice_colors):
        ax1.axvline(x=qt, color=color, linestyle=':', linewidth=1.5, alpha=0.9)
        # Place label text at the top of the line
        ax1.text(qt, np.max(omega) * 1.01, f'$q={qt}$', color=color, ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    ax1.grid(False) # Disable grid on the 2D heatmap to prevent visual artifacts
    ax1.set_xlabel(r"Vecteur d'onde normalisé $qa/\pi$", fontweight='bold')
    ax1.set_ylabel(r'Fréquence $\omega$ (u.a.)', fontweight='bold')
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, np.max(omega) * 1.05)
    
    # Fully opaque white legend with black border and black text (inverted colors)
    legend = ax1.legend(loc='lower right', facecolor='white', edgecolor='black', framealpha=1.0)
    
    # Colorbar
    cbar = fig.colorbar(heatmap, ax=ax1, pad=0.02)
    cbar.set_label('Intensité Spectrale Normalisée', rotation=270, labelpad=15, fontweight='bold')
    
    # --- PANEL 2: 1D Slices Centered on the Ballistic Line ---
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Ligne balistique')
    
    for idx, qt, color in zip(indices, q_targets, slice_colors):
        w_bal = 2.0 * np.sin(qt * np.pi / 2)
        delta_w = omega - w_bal
        intensity = psd_norm[idx, :]
        
        # Raw data (very low opacity)
        ax2.plot(delta_w, intensity, color=color, alpha=0.15, linewidth=0.8)
        
        # Smoothed data (Gaussian filter)
        smoothed = gaussian_filter1d(intensity, sigma=5.0)
        ax2.plot(delta_w, smoothed, color=color, alpha=1.0, linewidth=2.5, label=f'$q = {qt}$')
        
    ax2.set_xlabel(r'Désaccord de fréquence $\Delta\omega = \omega - \omega_{\mathrm{bal}}(q)$ (u.a.)', fontweight='bold')
    ax2.set_ylabel(r'Intensité spectrale normalisée', fontweight='bold')
    ax2.set_xlim(-0.6, 0.6) # Focused view around the peak detuning
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    pdf_path = os.path.join(fig_dir, "11_spectral_broadening.pdf")
    plt.savefig(pdf_path, dpi=300)
    print(f"--> Generated: {pdf_path}")
if __name__ == '__main__':
    print('Running plot_spectral_broadening...')
    plot_spectral_broadening()
    print('Done.')
