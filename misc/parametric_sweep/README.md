# Analyse — Étude Paramétrique de la Longueur de Pénétration

Ce répertoire contient une suite d'analyses numériques orientées objet
permettant d'étudier systématiquement l'influence de différents paramètres
physiques sur la **longueur de pénétration** ($L_p$) dans
les chaînes harmoniques 1D désordonnées.

## Structure

```
Analyse/
├── README.md
├── run_all.py              # Script maître : lance toutes les analyses
├── engine/                 # Moteur physique orienté objet
│   ├── __init__.py
│   ├── chain.py            # Classe DisorderedChain (modèle de paquet d'onde Verlet)
│   └── style.py            # Configuration esthétique LaTeX (cohérente avec le rapport)
├── scripts/                # Scripts d'analyse paramétrique
│   ├── analyse_01_disorder_type.py      # Effet du type de désordre (raideur/masse/combiné)
│   ├── analyse_02_seed_convergence.py   # Convergence statistique (nombre de réalisations)
│   ├── analyse_03_excitation_type.py    # Paquet d'onde vs impact
│   ├── analyse_04_chain_length.py       # Effet de la taille de la chaîne
│   ├── analyse_05_penetration_sweep.py  # Balayage 2D : Fréquence vs Désordre
│   └── analyse_06_frequency_sweep.py    # Balayage fréquentiel haute performance JIT-Numba
├── data/                   # Fichiers CSV générés (e.g. enveloppes de balayage)
└── output/                 # Figures PDF générées (gitignored)
```

## Caractère Standalone / Autonome

Ce répertoire est **100% autonome**. Il ne contient aucun lien rigide ni dépendance par rapport au reste du projet. Vous pouvez le copier n'importe où sur votre machine et l'exécuter directement.

## Utilisation et Installation

### Option A : Installation standard (avec connexion Internet)
```bash
pip install -r requirements.txt
```

### Option B : Installation autonome/hors-ligne (depuis les wheels locaux de `libs/`)
Si vous exportez ce dossier vers un nouvel environnement ou une machine sans accès Internet, toutes les dépendances binaires précompilées (wheels) sont stockées localement dans le dossier `libs/`. Installez-les directement avec :
```bash
pip install --no-index --find-links=libs/ -r requirements.txt
```

## Lancement des Analyses

1. **Lancer toutes les analyses paramétriques (01 à 06)** :
   ```bash
   python run_all.py
   ```

2. **Lancer un script individuel** :
   ```bash
   python scripts/analyse_06_frequency_sweep.py
   ```

## Convention de Nommage des Figures

Toutes les figures sont générées au format **PDF vectoriel** (300 dpi)
et suivent la convention de nommage :
```
analyse_XX_<description>.pdf
```
