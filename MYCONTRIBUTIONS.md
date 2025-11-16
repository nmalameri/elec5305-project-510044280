# 0. Summary of Contribution to the Topic

This project makes a clear contribution to the field of deepfake speech detection by developing a fully reproducible, lightweight baseline framework for the ASVspoof 2019 Logical Access task. The implementation provides:

- transparent LFCC/MFCC front-end comparisons under controlled conditions,
- an attack-balanced GMM training pipeline not present in official baselines,
- a compact CNN architecture tailored for cepstral inputs,
- per-attack analysis tools providing fine-grained insight into model weaknesses,
- a unified, deterministic experimentation structure enabling fair side-by-side evaluation.

These additions extend beyond simple reproduction of known baselines. The framework offers a practical, CPU-friendly foundation for teaching, benchmarking, and future research on synthetic speech detection, while revealing empirical behaviours—such as CNN sensitivity to DEV→EVAL shift and attack-specific vulnerabilities—that deepen understanding of anti-spoofing systems.

---

# Project Contributions

This document summarises the specific technical and conceptual contributions made during the development of this project.  
All components of the end-to-end system were implemented within this repository for the purpose of this assignment.

---

# 1. End-to-End Pipeline Design (01 → 10)

A complete, modular pipeline was designed and implemented from scratch, covering:

- dataset verification  
- manifest generation  
- cepstral feature extraction  
- GMM and CNN model training  
- scoring and decision logic  
- ROC/EER analysis  
- per-attack evaluation  
- results aggregation  
- visualisation  
- inference demonstration

Each stage is encapsulated in a dedicated script under `tools/`, ensuring clarity, reproducibility, and ease of extension.

---

# 2. Feature Extraction: LFCC & MFCC Front Ends

Implemented full cepstral feature extraction pipelines:

- LFCC (linear-frequency) and MFCC (mel-frequency) front ends  
- 25 ms frame windows, 10 ms hop  
- 20 cepstra including c₀  
- Δ and Δ² dynamic features  
- energy flooring and log compression  
- global standardisation using only training statistics  
- efficient `.npz` feature storage with resume-safe processing

This provides a transparent, CPU-friendly feature framework aligned with ASVspoof literature.

---

# 3. GMM Baseline Implementation

Developed a reproducible GMM-based spoofing countermeasure:

- per-class diagonal-covariance Gaussian mixture models  
- EM training with safe caps and numerically stable updates  
- attack-balanced frame sampling to prevent bias  
- mean log-likelihood ratio scoring  
- DEV-set EER threshold estimation  
- integrated ROC, EER, accuracy, TPR/FPR computation  
- per-attack EER analysis

This forms a strong, interpretable baseline consistent with ASVspoof standards.

---

# 4. Lightweight CNN Baseline Development

Designed and implemented a compact 2D CNN architecture for cepstral maps:

- 3× Conv–BN–ReLU blocks (3×3 and 5×5 kernels)  
- 2× MaxPooling layers  
- Global average pooling over time–frequency  
- Single-logit output  
- weighted BCE loss to handle class imbalance  
- AdamW with cosine learning rate scheduling  
- central or random T=400-frame cropping  
- CPU/MPS-friendly operation

Evaluation includes ROC curves, EER, per-attack metrics, and score distributions.

---

# 5. Results Aggregation & Analysis Tools

Created a suite of reproducibility and analysis utilities:

- automatic ROC CSV generation (DEV + EVAL)  
- summary table generation (CSV + LaTeX)  
- per-attack EER merging across models  
- automated visualisation of ROC overlays  
- unified comparison tool for LFCC/MFCC × GMM/CNN  
- secure run-directory structure storing configs, scalers, weights, scores, metrics, and ROC samples

This enables inspection, auditing, and fair model comparison.

---

# 6. Inference Demonstration Script

Implemented `tools/10_demo_examples.py` to provide a transparent demonstration of model inference:

- loads scores from saved experiments  
- applies DEV-set thresholding logic  
- evaluates both a spoof and a bona fide example  
- prints each model’s score, threshold, prediction, and correctness  
- ensures consistent comparison across all models  

This script supports public demonstration and supervisor-level inspection.

---

# 7. Documentation & Reproducibility Infrastructure

Produced comprehensive documentation:

- a detailed top-level `README.md`  
- dataset setup instructions  
- pipeline description  
- example input/output analysis  
- inference example tables  
- directory structure overview  
- reproducibility guarantees  
- separate file illustrating model predictions  
- this contributions file

All code is structured to allow complete reproduction of experiments via saved configurations and deterministic seeds.

---

# 8. Conceptual & Analytical Contributions

Beyond implementation, the project delivers:

- systematic comparison of LFCC vs MFCC  
- performance characterisation of GMM vs CNN  
- DEV→EVAL generalisation insights  
- identification of attack-specific weaknesses (A17, A19)  
- explanation of CNN misclassification behaviour  
- analysis of low-FPR performance differences  
- recommendations for future directions (CQCC, augmentation, fusion, robustness tests)

These analytical insights strengthen interpretation and academic contribution.

---

# 9. Overall Project Summary

The project includes:

- a **complete end-to-end deepfake detection pipeline**,  
- **carefully engineered baselines**,  
- **novel code written for this assignment**,  
- **clear explanations of model behaviour**, and  
- **professionally documented demonstration of inference outputs**.

This work forms a transparent, reproducible, and extendable baseline framework for ASVspoof-style research and teaching.
