"""
style.py — Configuration esthétique LaTeX pour les figures de rapport.

Extraite des conventions établies dans figures/02_linear_disorder/scripts/
pour garantir une cohérence visuelle parfaite avec le manuscrit principal.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt

# =============================================================================
# Palette de Couleurs (Premium Manuscript)
# =============================================================================
COLORS = {
    # Désordre de raideur (ressorts)
    'stiff':    '#d62728',   # Rouge professionnel
    # Désordre de masse
    'mass':     '#1f77b4',   # Bleu professionnel
    # Désordre combiné
    'combined': '#9467bd',   # Violet
    # Chaîne ordonnée / Référence
    'ordered':  '#2ca02c',   # Vert
    # Théorie / Fit analytique
    'theory':   '#000000',   # Noir
    # Accents secondaires
    'accent1':  '#ff7f0e',   # Orange
    'accent2':  '#17becf',   # Cyan
    'accent3':  '#8c564b',   # Brun
    # Fond de bande interdite
    'gap':      '#e0e0e0',   # Gris clair
}

# Labels LaTeX pour chaque type de désordre
DISORDER_LABELS = {
    'stiff':    r'D\'{e}sordre de raideur',
    'mass':     r'D\'{e}sordre de masse',
    'combined': r'D\'{e}sordre combin\'{e}',
}

def apply_style():
    """
    Applique la configuration matplotlib cohérente avec le rapport LaTeX.
    
    Convention héritée de figures/02_linear_disorder/scripts/:
    - text.usetex = True (rendu LaTeX natif)
    - font.family = serif (Times New Roman)
    - font.size = 13, axes.labelsize = 14, axes.titlesize = 16
    - grid.alpha = 0.2
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })
