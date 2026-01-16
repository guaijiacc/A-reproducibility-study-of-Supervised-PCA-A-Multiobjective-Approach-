# EECS 553 Course Project  
## Reproducibility Study of *Supervised PCA: A Multi-Objective Approach*

This repository contains the final report and MATLAB implementation for the EECS 553 (Machine Learning) course project at the University of Michigan.  
The project reproduces and evaluates the multi-objective supervised PCA method proposed in *Ritchie et al. (2020)*, with comparisons against Barshan’s SPCA and its kernelized variant.

---

## Project Overview

Supervised Principal Component Analysis (SPCA) aims to learn low-dimensional representations that are useful for prediction tasks. Many existing SPCA methods focus primarily on minimizing prediction error, often at the expense of variance explained, which can reduce interpretability.

This project studies a **multi-objective SPCA framework** that jointly optimizes:
- prediction accuracy, and
- variance explained in the original data.

We reproduce and compare the following methods:
- LSPCA (least-squares SPCA)
- LRPCA (logistic-regression PCA)
- Barshan’s SPCA
- Kernel Barshan SPCA (kBarshan)

Experiments are conducted on multiple regression and classification datasets to evaluate predictive performance and interpretability.

---

## Repository Structure

    .
    ├── EECS_553_Project_Report_Fall24.pdf                        # Final project report
    │
    ├── papers                                                    # Literature reviews
    │
    ├── code/data_preprocessing/                                  # Data loading, normalization, and splitting
    │
    ├── code/Barshan&kBarshan_method/                             # Barshan SPCA and kernel Barshan SPCA
    │
    ├── code/LSPCA&LRPCA/                                         # Multi-objective SPCA (LSPCA / LRPCA)
    │
    └── README.md

---

## Methods Implemented

### Multi-Objective SPCA

- **LSPCA**: Supervised PCA with least-squares regression loss  
- **LRPCA**: Supervised PCA with logistic regression loss  

Both methods use an alternating optimization scheme:
- Update regression coefficients using closed-form (linear) or built-in logistic regression
- Update the projection matrix using manifold optimization on the Stiefel manifold

Hyperparameters are selected using either:
- maximum likelihood estimation (MLE), or
- 10-fold cross-validation.

### Baseline Methods

- **Barshan’s SPCA** (HSIC-based dependence maximization)
- **Kernel Barshan SPCA (kBarshan)** using an RBF kernel

---

## Datasets

Experiments are performed on six datasets:

| Dataset       | Task           | Source |
|---------------|----------------|--------|
| Residential   | Regression     | UCI    |
| Music         | Regression     | UCI    |
| Barshan A     | Regression     | Synthetic |
| Ionosphere   | Classification | UCI    |
| Colon         | Classification | ASU    |
| Arcene        | Classification | ASU    |

For each dataset:
- 80% of data is used for training
- 20% is held out for testing
- 10-fold cross-validation is applied where needed
- The full procedure is repeated 10 times

---

## Requirements

- MATLAB
- Manopt toolbox for optimization on manifolds  
  https://www.manopt.org/

Some experiments (e.g., Arcene with cross-validation over subspace dimension) are computationally expensive and were run on a high-performance computing cluster.

---

## How to Run

### 1. Preprocess the data

    cd code_data_preprocessing
    run preprocess_<dataset>.m

### 2. Run Barshan / kernel Barshan

    cd code_Barshan&kBarshan_method
    run barshan_main.m

### 3. Run LSPCA / LRPCA

    cd code_LSPCA&LRPCA
    run lspca_main.m    % regression
    run lrpca_main.m    % classification

Hyperparameter selection (MLE vs cross-validation) can be configured in the main scripts.

---

## Results Summary

- LSPCA with cross-validation generally achieves the lowest regression error.
- Kernel Barshan often yields the lowest classification error.
- Multi-objective SPCA provides a better balance between prediction performance and variance explained.
- Selecting the subspace dimension using cross-validation consistently improves performance.

Detailed quantitative results and visualizations are provided in the project report.

---

## Contributions

- **Bobo Bai**  
  Barshan and kBarshan implementation, LSPCA/LRPCA reproduction, data preprocessing (Arcene and Barshan A), theoretical derivations, and report writing.

- **Shanchen Liu**  
  LRPCA implementation, data preprocessing (Ionosphere and Colon), and algorithm description.

- **Peng Zhai**  
  LSPCA implementation, data preprocessing (Residential and Music), experiments, figures, and high-performance computing support.

---

## References

1. A. Ritchie et al., *Supervised PCA: A Multi-Objective Approach*, arXiv:2011.05309, 2020  
2. E. Barshan et al., *Pattern Recognition*, 2011  
3. E. Bair et al., *Journal of the American Statistical Association*, 2006  
4. N. Boumal et al., *Manopt: A MATLAB Toolbox for Optimization on Manifolds*, JMLR, 2014  

---

## License

This repository is intended for educational and research purposes only.
