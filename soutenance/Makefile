# --- LaTeX Makefile (Soutenance) ---

LATEXMK = $(shell which latexmk 2>/dev/null || echo /Library/TeX/texbin/latexmk)
LUALATEX = $(shell which lualatex 2>/dev/null || echo /Library/TeX/texbin/lualatex)
BIBER = $(shell which biber 2>/dev/null || echo /Library/TeX/texbin/biber)

.PHONY: all clean fast

all: assets/logo-ups.pdf assets/lms-logo.pdf
	@echo "Building Slides (Full Compile with latexmk)..."
	mkdir -p build
	$(LATEXMK) -pdf -lualatex main.tex
	python3 script/create_pdfpc_from_notes.py
	cp build/main.pdf ./main.pdf
	cp build/main.pdf ./Soutenance-MELLET.pdf
	cp build/main.synctex.gz ./main.synctex.gz
	cp build/main.synctex.gz ./Soutenance-MELLET.synctex.gz
	@echo "PDF delivered to: main.pdf and Soutenance-MELLET.pdf"

assets/logo-ups.pdf:
	@if [ -f assets/logo-ups.svg ]; then \
		echo "Converting logo-ups.svg to PDF..."; \
		rsvg-convert -f pdf -o assets/logo-ups.pdf assets/logo-ups.svg; \
	fi

assets/lms-logo.pdf:
	@if [ -f assets/lms-logo.png ]; then \
		echo "Converting lms-logo.png to PDF..."; \
		inkscape assets/lms-logo.png --export-filename=assets/lms-logo.pdf; \
	fi

fast: assets/logo-ups.pdf assets/lms-logo.pdf
	@echo "Quick Recompile (lualatex)..."
	mkdir -p build
	$(LUALATEX) -synctex=1 -interaction=nonstopmode -output-directory=build main.tex
	python3 script/create_pdfpc_from_notes.py
	cp build/main.pdf ./main.pdf
	cp build/main.pdf ./Soutenance-MELLET.pdf
	cp build/main.synctex.gz ./main.synctex.gz
	cp build/main.synctex.gz ./Soutenance-MELLET.synctex.gz

clean:
	@echo "Cleaning auxiliary files..."
	$(LATEXMK) -c -output-directory=build
	rm -rf build/
	rm -f main.pdf Soutenance-MELLET.pdf main.synctex.gz Soutenance-MELLET.synctex.gz assets/logo-ups.pdf assets/lms-logo.pdf
