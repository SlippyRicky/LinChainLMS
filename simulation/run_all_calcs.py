"""
run_all_calcs.py
================
The master execution script for the entire linear disorder pipeline.
Running this script sequentially executes `sim_01` through `sim_07`,
repopulating the `data/` directory with fresh physical calculations.
"""
import os
import subprocess
import time
import sys

scripts = [
    'sim_01_disorder_baselines_audit.py',
    'sim_02_eigenmodes_dos_calculations.py',
    'sim_03_attenuation_theory_numerical_verification.py',
    'sim_04_wavepacket_spatiotemporal_propagation.py',
    'sim_05_wavepacket_localization_length_extraction.py',
    'sim_06_wavepacket_envelope_profiles.py',
    'sim_07_intermediate_scattering_dynamics.py'
]

if __name__ == '__main__':
    import sim_config
    data_dir = sim_config.DATA_DIR
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=================================================================")
    print("   MASTER CALCULATION PIPELINE                                   ")
    print("=================================================================")
    print(f"Target data directory: {data_dir}")
    print(f"Accuracy mode: {os.environ.get('SIM_ACCURACY_MODE', sim_config.ACCURACY_MODE)}")
    
    # Clean up generated files (keep static benchmark/metadata files)
    print("Cleaning generated physical datasets...")
    for f in os.listdir(data_dir):
        if f not in ('hu_2008_localization.csv', 'sim_metadata.csv', 'eps_metadata_stiffness.csv'):
            full_path = os.path.join(data_dir, f)
            if os.path.isfile(full_path):
                try:
                    os.remove(full_path)
                except Exception as e:
                    print(f"Error removing {f}: {e}")

    t0_global = time.time()
    for script in scripts:
        print(f'\n>>> Launching {script}...')
        script_path = os.path.join(script_dir, script)
        result = subprocess.run([sys.executable, script_path], cwd=script_dir)
        if result.returncode != 0:
            print(f"\n[ERROR] {script} failed! Stopping pipeline.")
            sys.exit(1)
            
    print("\n=================================================================")
    print(f"   ALL CALCULATIONS COMPLETE IN {time.time() - t0_global:.1f}s! ")
    print("=================================================================")
