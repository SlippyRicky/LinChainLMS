import os
import matplotlib.pyplot as plt

# Dynamic Path Resolution
PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PLOT_DIR, ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIGURES_ROOT = os.path.join(REPO_ROOT, "report", "figures")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_ROOT, exist_ok=True)

# Central styling function that scripts can call or is applied on import
def apply_style(font_size=18, linewidth=2.0, use_grid=False, grid_alpha=0.2):
    """
    Applies the default premium styling preset for the report figures.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": font_size,
        "axes.labelsize": font_size + 2,
        "axes.titlesize": font_size + 3,
        "legend.fontsize": font_size - 3,
        "lines.linewidth": linewidth,
        "axes.grid": use_grid,
        "grid.alpha": grid_alpha,
        "grid.linestyle": "--"
    })

# Apply default style on import (scripts can override afterward)
apply_style()

import matplotlib.figure
original_figure_savefig = matplotlib.figure.Figure.savefig

def save_figure(name, fig=None, category=None, **kwargs):
    """
    Saves the matplotlib figure to the appropriate destination directory
    based on the globally selected format ('pdf', 'png', or 'both').
    """
    if fig is None:
        fig = plt.gcf()
        
    # 1. Determine target directory
    target_dir = FIGURES_ROOT
    if category:
        target_dir = os.path.join(FIGURES_ROOT, category)
    os.makedirs(target_dir, exist_ok=True)
    
    # 2. Get the requested format from environment variable (default: pdf)
    fmt = os.environ.get("FIGURE_FORMAT", "pdf").strip().lower()
    
    saved_paths = []
    
    # 3. Save as PDF if requested or if 'both' is selected
    if fmt in ("pdf", "both"):
        save_path = os.path.join(target_dir, f"{name}.pdf")
        pdf_kwargs = kwargs.copy()
        # Call original unpatched savefig to avoid infinite recursion
        original_figure_savefig(fig, save_path, **pdf_kwargs)
        saved_paths.append(save_path)
        
    # 4. Save as PNG if requested or if 'both' is selected
    if fmt in ("png", "both"):
        save_path = os.path.join(target_dir, f"{name}.png")
        png_kwargs = kwargs.copy()
        # Default PNG to 300 dpi if not provided
        if "dpi" not in png_kwargs:
            png_kwargs["dpi"] = 300
        # Call original unpatched savefig to avoid infinite recursion
        original_figure_savefig(fig, save_path, **png_kwargs)
        saved_paths.append(save_path)
        
    # Print status message
    rel_paths = [os.path.relpath(p, REPO_ROOT) for p in saved_paths]
    print(f"--> Figure saved to: {', '.join(rel_paths)}")

# --- Monkey-patching matplotlib.figure.Figure.savefig ---
def custom_figure_savefig(self, fname, *args, **kwargs):
    """
    Intercepts all calls to fig.savefig, extracts the figure name,
    and reroutes it through our central save_figure method.
    """
    if isinstance(fname, str):
        basename = os.path.splitext(os.path.basename(fname))[0]
        
        # Check if this is a slides figure
        if "slides" in fname or "slides" in basename:
            category = "slides"
        else:
            category = "02_linear_disorder"
            
        save_figure(basename, fig=self, category=category, **kwargs)
    else:
        # Fallback to original savefig if fname is a file-like object
        original_figure_savefig(self, fname, *args, **kwargs)

matplotlib.figure.Figure.savefig = custom_figure_savefig
plt.savefig = lambda fname, *args, **kwargs: plt.gcf().savefig(fname, *args, **kwargs)