# Compilation options
PDFLATEX = lualatex --synctex=1 --interaction=nonstopmode
OUTPUT_DIR = build

.PHONY: all clean view

all:
	mkdir -p $(OUTPUT_DIR)
	$(PDFLATEX) --output-directory=$(OUTPUT_DIR) annexe_physique.tex
	$(PDFLATEX) --output-directory=$(OUTPUT_DIR) annexe_physique.tex
	$(PDFLATEX) --output-directory=$(OUTPUT_DIR) script_oral.tex
	$(PDFLATEX) --output-directory=$(OUTPUT_DIR) script_oral.tex
	cp $(OUTPUT_DIR)/annexe_physique.pdf .
	cp $(OUTPUT_DIR)/script_oral.pdf .

clean:
	rm -rf $(OUTPUT_DIR)
	rm -f annexe_physique.pdf script_oral.pdf
	rm -f *.synctex.gz
