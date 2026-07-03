import os
import subprocess
import glob

scripts = [
    # Main Report Figures (Chronological)
    "plot_01_historical_context.py",
    "plot_02_dispersion.py",
    "plot_03_diatomic_modes.py",
    "plot_04_disordered_chain_schematic.py",
    "plot_05_spectral_asymmetry.py",
    "plot_06_transfer_flowchart.py",
    "plot_07_eigenmodes_waterfall.py",
    "plot_10_participation_ratio.py",
    "plot_08_wavepacket_envelopes.py",
    "plot_11_spectral_broadening.py",
    "plot_12_matsuda_ishii.py",
    "plot_09_wp_vs_impact.py",
    "plot_13_attenuation_methods_comparison.py",
    "plot_14_scaling_fits_illustration.py",
    "plot_15_parameters_comparison.py",
    "plot_16_localization_phase_diagram.py",
    "plot_17_intermediate_scattering.py",
    "plot_09_wavepacket_fit_parameters.py",
]

if __name__ == '__main__':
    # --- CLEANING STEP ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, "../.."))
    print(f"Cleaning PDF and PNG outputs in: {output_dir}")
    if os.path.exists(output_dir):
        files = glob.glob(os.path.join(output_dir, "[0-9][0-9]_*.pdf")) + \
                glob.glob(os.path.join(output_dir, "[0-9][0-9]_*.png")) + \
                glob.glob(os.path.join(output_dir, "OLD_*.pdf")) + \
                glob.glob(os.path.join(output_dir, "OLD_*.png"))
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Error removing {f}: {e}")

    failed_scripts = []
    import sys
    for script in scripts:
        print(f'\n--- Running {script} ---')
        script_path = os.path.join(script_dir, script)
        result = subprocess.run([sys.executable, script_path])
        if result.returncode != 0:
            failed_scripts.append(script)

    print("\n" + "="*40)
    if not failed_scripts:
        print("[SUCCESS] All figures generated successfully.")
    else:
        print(f"[WARNING] {len(failed_scripts)} scripts failed to run:")
        for script in failed_scripts:
            print(f"  - {script}")
    print("="*40)
