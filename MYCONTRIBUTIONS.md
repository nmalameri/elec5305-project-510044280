# 0. Summary of Contribution to the Topic

This project makes a clear contribution to the field of deepfake speech detection by developing a **fully reproducible, lightweight baseline framework** for the ASVspoof 2019 Logical Access (LA) task. The framework provides:

- transparent LFCC/MFCC front-end comparisons under controlled conditions,  
- an attack-balanced GMM training pipeline not present in official baselines,  
- a compact CNN architecture tailored for cepstral inputs,  
- per-attack analysis tools that reveal fine-grained model behaviour,  
- a unified and deterministic experimentation environment enabling fair comparison across models,  
- optional integration of modern pretrained waveform models (RawNetLite, RawNet2, AASIST-LA).  

These additions go beyond simple reproduction of the ASVspoof baseline.  
The system offers a **clean, CPU-friendly research platform** for teaching and benchmarking, while the analyses highlight empirical behaviours—such as CNN sensitivity to DEV→EVAL shift, attack-specific vulnerabilities, and the effect of naive pretrained model usage without matched preprocessing—that deepen understanding of anti-spoof generalisation.

---

# Project Contributions

This document summarises the **technical** and **conceptual** contributions made during the development of this project.  
All components of the end-to-end system were implemented within this repository for the purpose of this assignment.

---

# 1. End-to-End Pipeline Design (01 → 10)

A complete, modular, and reproducible pipeline was designed and implemented from scratch covering:

- dataset verification  
- manifest generation  
- cepstral feature extraction  
- GMM and CNN training  
- scoring and thresholding logic  
- ROC/EER analysis  
- per-attack evaluation (A07–A19)  
- cross-model results aggregation  
- visualisation  
- inference demonstration  

Each stage is encapsulated as a dedicated script under `tools/`, ensuring clarity, maintainability, and extensibility.

---

# 2. Feature Extraction: LFCC & MFCC Front Ends

Implemented full LFCC/MFCC front ends including:

- LFCC (linear-frequency) and MFCC (mel-frequency) spectral features  
- 25 ms frames, 10 ms hop  
- 20 cepstra including c₀  
- Δ and Δ² dynamic features (full 60-dim vector)  
- energy flooring, log compression  
- global standardisation using training-only statistics  
- efficient `.npz` feature storage with resume-safe behaviour  

This provides an interpretable and CPU-efficient feature framework aligned with ASVspoof literature.

---

# 3. GMM Baseline Implementation

Developed a reproducible LFCC/MFCC GMM countermeasure including:

- per-class diagonal GMMs  
- EM training with stable updates  
- attack-balanced sampling to prevent bias  
- capped frame pooling to avoid domination by long utterances  
- mean log-likelihood ratio scoring  
- DEV-set EER threshold estimation  
- ROC/EER/ACC computation  
- per-attack EER evaluation and LaTeX-ready tables  

This results in a strong, interpretable baseline consistent with challenge protocols.

---

# 4. Lightweight CNN Baseline Development

Designed and implemented a compact CNN model for cepstral maps:

- three Conv–BN–ReLU blocks using 3×3 and 5×5 filters  
- two max-pool stages  
- global average pooling across time–frequency  
- single-logit countermeasure output  
- weighted BCE loss for class imbalance  
- AdamW with cosine scheduling  
- central/random 400-frame cropping  
- MPS/CPU friendly code paths  

Evaluation includes DEV/EVAL ROC curves, EERs, score distributions, and per-attack error tables.

---

# 5. Results Aggregation & Analysis Tools

Implemented a suite of utilities to consolidate results:

- automatic ROC CSV generation (DEV + EVAL)  
- headline table generation (CSV + Markdown + LaTeX)  
- per-attack EER merging across feature and model types  
- unified ROC overlays  
- reproducible run-directory structures containing configs, scalers, weights, scores, and ROC samples  

This enables deeper inspection, auditing, and fair comparison across all systems.

---

# 6. Inference Demonstration Script

Created `tools/10_demo_examples.py` to provide an intuitive, inspectable demonstration of model behaviour:

- loads scores from saved runs  
- applies the DEV-set threshold consistently  
- evaluates both a spoof and a bona fide example  
- prints per-model scores, decisions, and correctness  
- reveals feature/model sensitivity on the same input  

This script supports public demonstration, debugging, and understanding of classifier decisions.

---

# 7. Pretrained Model Integration (RawNetLite, RawNet2, AASIST-LA)

Integrated inference wrappers for three pretrained anti-spoofing systems:

- **RawNetLite** (waveform model)
- **RawNet2** (deep raw-audio GRU model)
- **AASIST-LA** (graph attention model for ASVspoof 2021 LA)

Each wrapper:

- loads pretrained weights,  
- performs waveform preprocessing (cropping, padding, normalisation),  
- extracts model-specific logits,  
- produces `scores_dev.csv`, `scores_eval.csv`, `metrics.json`, and `run_config.json`.  

A standalone per-attack evaluation tool was implemented to compute:

- `per_attack_eer_eval.csv`  
- `per_attack_op_eval_at_dev_threshold.csv`  

**Important Insight:**  
Since these models originate from external repositories with specialised preprocessing (pre-emphasis, sinc filters, mel parameters, graph attention frontends), naïve integration (without reimplementing the full original pipeline) results in **domain/feature mismatch**. This manifests as inverted score distributions and ROC curves below the random baseline.

This is reflected explicitly in the analysis rather than hidden, turning pretrained mismatch into a clear empirical finding about anti-spoof generalisation.

---

# 8. Unified Cross-Model Summary Tools

Added `tools/11_collect_all_results.py` to aggregate:

- GMM (LFCC/MFCC)
- CNN (LFCC/MFCC)
- RawNetLite (pretrained)
- RawNet2 (pretrained)
- AASIST-LA (pretrained)

into single summary outputs:

- `headline_all_models.csv`  
- `headline_all_models.md`  
- `roc_eval_all_models.png`  

This provides a complete side-by-side comparison across all baselines and pretrained models.

---

# 9. Documentation & Reproducibility Infrastructure

Produced extensive documentation:

- a polished `README.md` explaining installation, dataset setup, pipeline usage, and examples  
- a separate example-inference file demonstrating per-model decisions  
- detailed contributions and analysis files  
- consistent structure for all run directories  

Reproducibility was ensured via:

- deterministic seeds  
- preserved run configurations  
- consistent scoring formats  
- strict separation of DEV thresholding and EVAL scoring.

---

# 10. Conceptual & Analytical Contributions

Beyond engineering, the project contributes:

- systematic exploration of LFCC vs MFCC behaviour  
- analysis of GMM vs CNN generalisation  
- identification of hardest attacks (A17, A19)  
- explanation for CNN DEV overfitting  
- interpretation of ROC operating points  
- empirical demonstration of domain mismatch in pretrained raw-audio models  
- insights into the limits of raw-audio systems without proper preprocessing  
- recommendations for future work (CQCCs, augmentation, embedding fusion, robust training)