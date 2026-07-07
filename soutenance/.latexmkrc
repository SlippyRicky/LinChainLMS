# --- latexmk configuration for Soutenance (Slides) ---
# Put all output and auxiliary files in the build directory
$out_dir = 'build';

# Use LuaLaTeX (pdf_mode = 4)
$pdf_mode = 4;
$lualatex = 'lualatex -synctex=1 -interaction=nonstopmode %O %S';

# Define extensions to clean up
$clean_ext = 'bcf bbl run.xml synctex.gz nav snm';
