"""
run_all_calcs.py
================
The master execution script for the entire linear disorder pipeline.
Running this script sequentially executes `calc_01` through `calc_06`,
repopulating the `data/` directory with fresh calculations.
"""
import os
import subprocess
import glob
import time

scripts = [
    'sim_01_disorder_baselines_audit.py',
    'sim_02_eigenmodes_dos_calculations.py',
    'sim_03_attenuation_theory_numerical_verification.py',
    'sim_04_wavepacket_spatiotemporal_propagation.py',
    'sim_05_wavepacket_localization_length_extraction.py',
    'sim_06_wavepacket_envelope_profiles.py'
]

if __name__ == '__main__':
    import sim_config
    import sys
    data_dir = sim_config.DATA_DIR
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=================================================================")
    print("   MASTER CALCULATION PIPELINE                                   ")
    print("=================================================================")
    print(f"Target data directory: {data_dir}")
    
    # Clean up generated files (keep static benchmark/metadata files)
    print("Cleaning generated physical datasets...")
    for f in os.listdir(data_dir):
        if f not in ('hu_2008_localization.csv', 'sim_metadata.csv', 'eps_metadata_stiffness.csv'):
            try:
                os.remove(os.path.join(data_dir, f))
            except Exception as e:
                print(f"Error removing {f}: {e}")


    t0_global = time.time()
    for script in scripts:
        print(f'\n>>> Launching {script}...')
        script_path = os.path.join(script_dir, script)
        result = subprocess.run(['python3', script_path])
        if result.returncode != 0:
            print(f"\n[ERROR] {script} failed! Stopping pipeline.")
            break
            
    print("\n=================================================================")
    print(f"   ALL CALCULATIONS COMPLETE IN {time.time() - t0_global:.1f}s! ")
    print("=================================================================")
