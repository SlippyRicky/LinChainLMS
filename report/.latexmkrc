# --- latexmk configuration ---

# Directories
$out_dir = 'build';
$aux_dir = 'build';

# Engines
$pdf_mode = 4; # lualatex
$post_processor = 'biber --input-directory=build --output-directory=build %S';
$lualatex = 'lualatex -interaction=nonstopmode -shell-escape -synctex=1 -output-directory=build %O %S';

# Cleanup extra files
@generated_exts = (@generated_exts, 'run.xml', 'bcf', 'synctex.gz');
