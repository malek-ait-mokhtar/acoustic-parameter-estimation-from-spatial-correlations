# Acoustic Parameter Estimation from Spatial Correlations

**Passive estimation of acoustic wave parameters from microphone-array measurements using spatial covariance models and Gaussian processes.**

---

## Overview

This project investigates the passive estimation of acoustic wave parameters in reverberant environments.

Under the diffuse-field assumption, the spatial correlation between pressure measurements follows the theoretical model

[
R(r)=\frac{\sin(kr)}{kr},
]

where (k) denotes the acoustic wavenumber.

Using real measurements collected with microphone arrays, the project estimates the acoustic wavenumber and the speed of sound through nonlinear optimization of spatial correlation models.

The study also explores Gaussian Process (GP) regression for sound field reconstruction and uncertainty quantification.

---

## Key Contributions

* Developed a passive estimator of acoustic wave speed from spatial correlation measurements.
* Implemented a complete signal-processing pipeline based on Cross-Spectral Matrices (CSM) and spatial coherence estimation.
* Investigated the impact of microphone-array geometry on estimation performance.
* Analyzed the multimodal structure of the nonlinear least-squares objective function.
* Explored alternative estimators based on local-minimum analysis and segmented affine fitting.
* Implemented Gaussian Process regression for sound field reconstruction and uncertainty quantification.
* Validated the methodology on real experimental data collected with 16-channel and 24-channel microphone arrays.

---

## Methodology

### Signal Processing

* Multi-channel microphone-array acquisition
* Welch spectral estimation
* Cross-Spectral Matrix (CSM) computation
* Spatial coherence estimation

### Statistical Inference

* Nonlinear least-squares estimation of the acoustic wavenumber
* Objective-function analysis
* Local-minima investigation
* Confidence interval estimation

### Gaussian Processes

* Diffuse-field covariance kernel design
* Sound field reconstruction
* Predictive uncertainty quantification
* Leave-One-Out validation

---

## Experimental Setup

### UMA16 Planar Array

* 16 microphones
* Regular 4×4 geometry
* 4 cm spacing

### 3D Microphone Array

* 24 microphones
* Three orthogonal branches
* 239 distinct microphone-pair distances

The influence of array geometry on estimation accuracy was analyzed experimentally.

---

## Main Results

* Accurate sound-speed estimation within the frequency range where the diffuse-field model is valid.
* Identification of low-frequency limitations caused by modal room behavior.
* Analysis of high-frequency failures caused by objective-function multimodality.
* Improved estimation performance using a 3D microphone-array geometry.
* Successful Gaussian Process reconstruction of acoustic fields from sparse measurements.

---

## Repository Structure

```text
.
├── docs/
│   ├── report.pdf
│   └── presentation.pdf
│
├── src/
│   ├── acquisition/
│   ├── estimation/
│   ├── gaussian_process/
│   └── visualization/
│
├── figures/
│
├── requirements.txt
│
└── README.md
```

---

## References

1. Cook, R. K., Waterhouse, R. V., Berendt, R. D., Edelman, S., & Thompson, M. C. (1955). *Measurement of Correlation Coefficients in Reverberant Sound Fields*. Journal of the Acoustical Society of America.

2. Caviedes-Nozal, D., Riis, N. A. B., Heuchel, F. M., Brunskog, J., Gerstoft, P., & Fernandez-Grande, E. (2021). *Gaussian Processes for Sound Field Reconstruction*. Journal of the Acoustical Society of America.

3. Basak, S., Petit, S., Bect, J., & Vazquez, E. (2021). *Numerical Issues in Maximum Likelihood Parameter Estimation for Gaussian Process Interpolation*.

---

## Authors

* Malek Aït-Mokhtar
* Malo Bonnefoy
* Antoine Bonnin
* Abdessellam Garrou
* Antoine Maillet

Project supervised by **Gilles Chardon**

Université Paris-Saclay · CentraleSupélec · Laboratoire des Signaux et Systèmes (L2S)

