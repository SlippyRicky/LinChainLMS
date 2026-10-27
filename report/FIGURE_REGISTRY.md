# Figure Registry

This file tracks all figures used in the internship report (`report/main.tex`) and presentations to prevent numbering collisions and ensure clear provenance.

## Main Report Figures (`report/main.tex`)
| ID | Script | Output Filename | Section / Topic | Status in `main.tex` |
|:--:|:-------|:----------------|:----------------|:---------------------|
| 01 | `plot_01_historical_context.py` | `01_historical_context.pdf` | Introduction: 3D ultrasound data & glass model | Integrated (line 166) |
| 02 | `plot_02_dispersion.py` | `02_dispersion.pdf` | Pristine monoatomic chain dispersion & DOS | Integrated (line 220) |
| 04 | `plot_04_disordered_chain_schematic.py` | `04_disordered_chain_schematic.pdf` | Linear chain under mass and stiffness disorder | Integrated (line 302) |
| 05 | `plot_05_spectral_asymmetry.py` | `05_spectral_asymmetry.pdf` | Spectral DOS asymmetry under disorder | Integrated (line 354) |
| 06 | `plot_06_transfer_flowchart.py` | `06_transfer_flowchart.pdf` | Transfer Matrix Method (TMM) algorithm flowchart | Integrated (line 248) |
| 07 | `plot_07_eigenmodes_waterfall.py` | `07_eigenmodes_waterfall.pdf` | Spatial profile of vibrational eigenmodes | Integrated (line 400) |
| 08 | `plot_08_wavepacket_envelopes.py` | `08_wavepacket_envelopes.pdf` | Time-domain wavepacket attenuation envelopes | Integrated (line 574) |
| 09 | `plot_09_wp_vs_impact.py` | `09_wp_vs_impact.pdf` | Spatio-temporal maps: Wavepacket vs Delta Impact | Integrated (line 630) |
| 10 | `plot_10_participation_ratio.py` | `10_participation_ratio.pdf` | Participation Ratio ($PR$) spectrum & acoustic plateau | Integrated (line 460) |
| 11 | `plot_11_spectral_broadening.py` | `11_spectral_broadening.pdf` | Spectral function broadening $S(q, \omega)$ & linewidths | Integrated (line 489) |
| 12 | `plot_12_matsuda_ishii.py` | `12_matsuda_ishii.pdf` | Matsuda-Ishii $\omega^2 \sigma^2$ scaling validation | Integrated (line 502) |
| 13 | `plot_13_attenuation_methods_comparison.py` | `13_attenuation_methods_comparison.pdf` | Multi-method attenuation comparison (TMM, Riccati, PR, MD) | Integrated (line 551) |
| 14 | `plot_14_scaling_fits_illustration.py` | `14_scaling_fits_illustration.pdf` | Power-law scaling fit methodology | Integrated (line 524) |
| 15 | `plot_15_parameters_comparison.py` | `15_parameters_comparison.pdf` | Scaling exponents across mass, stiffness, and combined disorder | Integrated (line 531) |

## Supplementary / Extended Study Figures
| ID | Script | Output Filename | Description | Usage |
|:--:|:-------|:----------------|:------------|:------|
| 03 | `plot_03_diatomic_modes.py` | `03_diatomic_modes.pdf` | Diatomic acoustic and optical branches & bandgap | Supplementary / Annex |
| 16 | `plot_16_localization_phase_diagram.py` | `16_localization_phase_diagram.pdf` | 2D phase maps of localization length $\ell(\omega, \varepsilon)$ | Supplementary / Annex |
| 17 | `plot_17_intermediate_scattering.py` | `17_intermediate_scattering.pdf` | Intermediate scattering function $F(q, t)$ from thermal MD | Supplementary / Annex |
| 20 | `plot_20_wavepacket_fit_parameters.py` | `20_wavepacket_fit_parameters.pdf` | Fitted wavepacket parameters ($\xi_E$, $\alpha$, $n_c$) vs frequency | Supplementary / Annex |

## Presentation Figures (`soutenance/figures/`)
| Script | Output Filename | Slide Description |
|:-------|:----------------|:------------------|
| `plot_01_participation_ratio.py` | `10_participation_ratio_slides.png` | Highlighted acoustic & localized regimes for slides |
| `plot_02_wavepacket_envelopes.py` | `08_wavepacket_envelopes_slides.png` | Wavepacket decay for presentation |
| `plot_03_attenuation_methods_comparison.py` | `13_attenuation_methods_comparison_slides.png` | Tri-panel comparison for presentation |

---
*Note: Generated figures are placed in `report/figures/02_linear_disorder/` and `soutenance/figures/`.*
