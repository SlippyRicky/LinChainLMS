"""
run_all.py — Script Maître : Lance Toutes les Analyses Paramétriques.

Usage:
    python run_all.py          # Lance toutes les analyses
    python run_all.py 1 3      # Lance uniquement les analyses 01 et 03
"""
import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')

ANALYSES = {
    1: ('analyse_01_disorder_type.py',    'Effet du Type de Désordre'),
    2: ('analyse_02_seed_convergence.py',  'Convergence Statistique (Seeds)'),
    3: ('analyse_03_excitation_type.py',   'Paquet d\'Onde vs Impact'),
    4: ('analyse_04_chain_length.py',      'Effet de la Taille de la Chaîne'),
    5: ('analyse_05_penetration_sweep.py',  'Longueur de Pénétration (Fréquence vs Désordre)'),
    6: ('analyse_06_frequency_sweep.py',    'Balayage Fréquentiel JIT-Numba (Masse vs Désordre)'),
}


def run_analysis(num, script, description):
    """Run a single analysis script and report timing."""
    path = os.path.join(SCRIPT_DIR, script)
    print(f"\n{'=' * 70}")
    print(f"  [{num:02d}] {description}")
    print(f"  Script : {script}")
    print(f"{'=' * 70}")
    
    t0 = time.time()
    result = subprocess.run([sys.executable, path],
                            cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0
    
    status = "✓ OK" if result.returncode == 0 else "✗ ERREUR"
    print(f"  [{status}] Terminé en {elapsed:.1f} secondes")
    return result.returncode == 0


def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║        ANALYSE PARAMÉTRIQUE — LONGUEUR DE LOCALISATION             ║")
    print("║        Chaînes Harmoniques 1D Désordonnées                         ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    # Parse which analyses to run
    if len(sys.argv) > 1:
        selected = [int(x) for x in sys.argv[1:]]
    else:
        selected = list(ANALYSES.keys())
    
    t_total = time.time()
    successes = 0
    failures = 0
    
    for num in selected:
        if num not in ANALYSES:
            print(f"\n  [!] Analyse {num} inconnue, ignorée.")
            continue
        script, desc = ANALYSES[num]
        ok = run_analysis(num, script, desc)
        if ok:
            successes += 1
        else:
            failures += 1
    
    elapsed_total = time.time() - t_total
    
    print(f"\n{'=' * 70}")
    print(f"  RÉSUMÉ : {successes} réussies, {failures} échouées")
    print(f"  Temps total : {elapsed_total:.1f} secondes")
    print(f"  Figures → Analyse/output/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
