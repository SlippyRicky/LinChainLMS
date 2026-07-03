import os
import sys
import subprocess
import argparse

# Dynamic root discovery
PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PLOT_DIR, ".."))

def run_pipeline(format_choice, skip_calcs):
    # 1. Set the global environment variable for all child scripts
    os.environ["FIGURE_FORMAT"] = format_choice
    print(f"Setting FIGURE_FORMAT = '{format_choice}'")

    # 2. Run simulation engine calculations if requested
    if not skip_calcs:
        print("\n>>> Running standalone simulation engine...")
        calc_script = os.path.join(REPO_ROOT, "simulation", "run_all_calcs.py")
        result = subprocess.run([sys.executable, calc_script], env=os.environ, cwd=os.path.dirname(calc_script))
        if result.returncode != 0:
            print("!!! Error executing simulation calculations. Exiting.")
            sys.exit(1)
    else:
        print("\n>>> Skipping simulation calculations. Using existing data.")

    # 3. Run plotting compilation pipeline
    print("\n>>> Running plotting compilation pipeline...")
    plot_script = os.path.join(REPO_ROOT, "plotting", "plot_all.py")
    result = subprocess.run([sys.executable, plot_script], env=os.environ, cwd=os.path.dirname(plot_script))
    if result.returncode != 0:
        print("!!! Error compiling figures. Exiting.")
        sys.exit(1)

    print("\n[SUCCESS] Pipeline completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate the linear chain simulation and figure generation pipeline.")
    parser.add_argument(
        "--format", "-f",
        choices=["pdf", "png", "both"],
        default="pdf",
        help="Figure output file format (default: pdf)"
    )
    parser.add_argument(
        "--skip-calcs", "-s",
        action="store_true",
        help="Skip running the simulation calculations and use existing cached datasets"
    )

    args = parser.parse_args()
    run_pipeline(args.format, args.skip_calcs)
