# --- Standalone Simulation & Report Makefile ---

.PHONY: all sim figs report soutenance clean clean-data fast

# Default target runs calculations, compiles figures, and builds the report and slides
all: sim figs report soutenance

# Run calculations using the simulation engine
sim:
	@echo "Running standalone simulation calculations..."
	python3 simulation/run_all_calcs.py

# Compile all figures (default format: pdf)
figs:
	@echo "Generating all project figures..."
	python3 plotting/plot_all.py

# Quick recompile of LaTeX report (skipping calculations/figures)
fast:
	@echo "Quick compiling LaTeX report..."
	cd report && latexmk -pdf -lualatex -synctex=1 -interaction=nonstopmode -shell-escape -outdir=build main.tex

# Full compilation of LaTeX report
report:
	@echo "Building LaTeX report..."
	cd report && latexmk -pdf -lualatex -synctex=1 -interaction=nonstopmode -shell-escape -outdir=build main.tex
	@echo "PDF delivered to: report/build/main.pdf"

# Full compilation of LaTeX presentation slides
soutenance:
	@echo "Building Soutenance slides..."
	cd soutenance && $(MAKE)

# Clean up all report compilation and figures
clean:
	@echo "Cleaning up LaTeX build files..."
	cd report && latexmk -C -outdir=build
	rm -rf report/build/
	rm -f report/*.aux report/*.log report/*.fls report/*.fdb_latexmk report/*.toc report/*.out report/*.lof report/*.blg report/*.bcf report/*.run.xml
	rm -f report/main.pdf
	@echo "Cleaning up Soutenance build files..."
	cd soutenance && $(MAKE) clean
	@echo "Cleaning up generated figures..."
	rm -rf report/figures/02_linear_disorder/*.pdf
	rm -rf report/figures/02_linear_disorder/*.png
	rm -rf report/figures/02_linear_disorder/slides/*.pdf
	rm -rf report/figures/02_linear_disorder/slides/*.png

# Clean up computed simulation datasets (keeping static datasets)
clean-data:
	@echo "Cleaning up calculated physical datasets..."
	find data -type f ! -name "hu_2008_localization.csv" ! -name "sim_metadata.csv" ! -name "eps_metadata_stiffness.csv" -delete
