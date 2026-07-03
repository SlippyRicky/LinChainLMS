# Linear Chain Simulation and Report Environment

This repository contains the standalone simulation engine, plotting scripts, and LaTeX report source code for the study of wave propagation and localization in one-dimensional disordered lattices.

## Repository Structure

The code is compartmentalized into four root-level directories to ensure modularity and ease of reuse:

* **`simulation/`**: The standalone simulation engine. These scripts perform exact diagonalization and time evolution simulations of 1D mass/stiffness disordered chains.
* **`data/`**: The central dataset store. Contains static metadata, literature benchmark files (e.g. Hu et al. 2008), and physical datasets computed by the simulation engine (NPZ and CSV formats).
* **`plotting/`**: The graphics compilation pipeline. Loads physical data from `data/` to compile report figures and presentation slides.
* **`report/`**: The LaTeX report workspace. Contains chapters, bibliography, and compiled PDF/PNG figures in `report/figures/`.

---

## Getting Started

### Prerequisites

Ensure you have a modern Python environment and a LaTeX distribution installed. You can install all Python dependencies using the provided `requirements.txt` file (compatible with macOS, Linux, and Windows):
```bash
pip install -r requirements.txt
```

For LaTeX compilation, **LuaLaTeX** is recommended:
```bash
# Mac (via Homebrew)
brew install --cask mactex-no-gui
# Or make sure latexmk and lualatex are available on your system path
```

---

## How to Run

The easiest way to orchestrate the environment is using the root **`Makefile`**:

### 1. Run Simulations
Computes all physical properties and populates the `data/` directory:
```bash
make sim
```
This runs the master calculation script `simulation/run_all_calcs.py`, which cleans up previous generated files (preserving static metadata/literature benchmarks) and runs all numerical calculations.

### 2. Generate Figures
Generates the figures for the report and slides:
```bash
make figs
```
This runs `plotting/plot_all.py` which compiles all figures and saves them in `report/figures/02_linear_disorder/`.

### 3. Build the Report
Compiles the LaTeX report:
```bash
make report
```
This runs `latexmk` inside the `report/` workspace and places the compiled document at `report/build/main.pdf`.

### 4. Build Everything (End-to-End)
To clean previous datasets/figures and run the entire pipeline end-to-end:
```bash
make clean
make clean-data
make
```

---

## Centralized Figure Output Options

You can easily configure the file type/format of the generated figures (supporting `pdf`, `png`, or `both`).

### Option A: Using the CLI Orchestrator
We provide a master runner script `plotting/make_all.py` that lets you choose format and skips calculations if desired:
```bash
# Generate PNG figures (saves to report/figures/...)
python3 plotting/make_all.py --format png

# Generate both PDF and PNG figures, skipping recalculations
python3 plotting/make_all.py --format both --skip-calcs
```

### Option B: Using Environment Variables
Individual plotting scripts check the `FIGURE_FORMAT` environment variable (defaults to `pdf`). This allows you to run single scripts manually and choose the format:
```bash
# Run a single script to generate a PNG figure
FIGURE_FORMAT=png python3 plotting/plot_02_dispersion.py

# Run a single script to generate both PDF and PNG formats
FIGURE_FORMAT=both python3 plotting/plot_02_dispersion.py
```

---

## Manual Wavepacket Fit Configuration

Some figures (such as the wavepacket envelopes in `plot_08_wavepacket_envelopes.py` and `plot_08_wavepacket_envelopes_slides.py`) rely on manual fit ranges to extract intermediate localization lengths:
* We provide an interactive utility `plotting/interactive_fit.py` that allows you to manually choose fit distances/ranges by hand using a graphical interface.
* Running this script writes your chosen ranges to `plotting/manual_fit_ranges.json`.
* **Fallback Behavior**: If `manual_fit_ranges.json` is missing or has not been generated, the `plot_08` scripts will automatically run without the manual fit data and plot only the raw data curves (without overlaying fit lines or crossover markers).

---

## Contribution & Guidelines
 
If you are modifying or extending this codebase:
* Always resolve folder paths relative to the repository root using `simulation/sim_config.py` and `plotting/plot_config.py`. Do **not** hardcode relative path jumps like `../../data`.
* Always use `plot_config.save_figure("filename", category="category_name")` in plotting scripts to preserve the centralized file type control.
* Check `.agents/AGENTS.md` for exhaustive workspace scripting guidelines.

---

## Citation

If you use this codebase or refer to the findings in the report, please cite it as:

```bibtex
@techreport{mellet2026waves,
  author      = {Mellet, Emeric},
  title       = {Ondes M{\'e}caniques dans les Chaines 1D D{\'e}sordonn{\'e}s},
  institution = {Laboratoire de M{\'e}canique des Solides (LMS), {\'E}cole Polytechnique},
  year        = {2026},
  type        = {Research Internship Report}
}
```

## AI Collaboration & Transparency Disclosure

This repository is the result of academic research conducted during a research internship. 
* **Core Research & Physics**: All physical modeling, mathematical derivations, analysis of the results, and research directions were conceived and directed by the human author (Emeric Mellet).
* **AI Assistance**: Large language models (LLMs) were utilized as pair-programming assistants to support code refactoring, styling optimization (Matplotlib), documentation, and codebase restructuring. 

## Disclaimer

The software and report are provided "as is", without warranty of any kind. The views and conclusions contained in the report are those of the author and should not be interpreted as representing the official policies or endorsements of the Laboratoire de Mécanique des Solides (LMS) or École Polytechnique.
