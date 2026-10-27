# Workspace Coding & Development Guidelines

These guidelines are project-specific rules to ensure consistency, clean structure, and ease of maintenance for both manual developers and AI agents working on this linear chain simulation and report repository.

## 1. Directory Structure Conventions

The repository is organized into four main root directories:
* `simulation/` - Standalone simulation engine scripts.
* `plotting/` - Plotting and figure generation scripts.
* `report/` - LaTeX report source code and assets.
* `data/` - Shared folder for generated physical datasets (NPZ, CSV).

No script should perform calculations or write output files outside this structure. Temporary files or deprecated scripts must not be checked in.

## 2. Python Scripting Rules

### A. Dynamic Path Resolution
* **NO hardcoded relative path jumps** (e.g. `../../data` or `../../../../code`) inside scripts.
* Simulation scripts must import `sim_config` and save their datasets to `sim_config.DATA_DIR`.
* Plotting scripts must import `plot_config` and resolve paths using:
  - `plot_config.DATA_DIR` for loading datasets.
  - `plot_config.save_figure(fig, "filename", category="category_name")` for saving plots.

### B. Matplotlib Styling
* Avoid hardcoding font sizes, styles, or grid alpha settings inside individual plotting scripts unless specifically overriding.
* Standard styling defaults are managed globally by importing `plot_config.py`.

### C. Script Headers
* Each new or modified python script must begin with a clear docstring header specifying:
  - **Purpose**: A brief description of what the script does.
  - **Inputs**: What datasets it reads from `data/`.
  - **Outputs**: What figures/data files it produces.

## 3. LaTeX Integration Rules

* Figures must be referenced in LaTeX using their filenames directly, capitalizing on the `\graphicspath` configuration in `report/config/preamble.tex`.
* **NO hardcoded subdirectory paths** (e.g. `\includegraphics{figures/02_linear_disorder/fig.pdf}`) should be used in the LaTeX sections. Use `\includegraphics{fig.pdf}` instead.
* Update `FIGURE_REGISTRY.md` when introducing new figures or modifying existing figure numbers.
