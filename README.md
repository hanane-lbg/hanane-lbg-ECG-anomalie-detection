# ECG Anomaly Detection

A comprehensive machine learning framework for detecting cardiac anomalies in electrocardiogram (ECG) signals using advanced signal processing and deep learning techniques.

---

## Abstract

This project implements an end-to-end pipeline for ECG-based cardiac anomaly detection, combining robust signal preprocessing with multiple deep learning paradigms. It integrates:

- **Unsupervised Autoencoders** for reconstruction-based anomaly scoring  
- **Supervised Sequence Models** (LSTM, Bi-LSTM, GRU, CNN-LSTM) for binary classification  

The framework achieves very high sensitivity in detecting pathological cardiac patterns while preserving specificity on normal signals, reaching up to **99.9% accuracy** with state-of-the-art architectures.

---

## Contents

1. [Overview](#overview)  
2. [System Architecture](#system-architecture)  
3. [Methodology](#methodology)  
4. [Project Structure](#project-structure)  
5. [Key Results](#key-results)  
6. [Design Rationale](#design-rationale)  
7. [Clinical Significance](#clinical-significance)  
8. [Technical Specifications](#technical-specifications)  
9. [Citation](#citation)

---

## Overview

ECG signals are noisy, highly variable, and morphologically complex. This project addresses these challenges through a two-stage pipeline:

- **Stage 1 – Preprocessing & Feature Engineering:**  
  Cleans raw ECG signals, normalizes them, and extracts clinically meaningful features.
- **Stage 2 – Deep Learning Models:**  
  Applies unsupervised and supervised models for anomaly detection.

The framework is modular, reproducible, and suitable for both research and clinical experimentation.

---

## System Architecture

### End-to-End Pipeline

![Image](./plots/complete_pipeline.png)

## Methodology

### 1. Signal Preprocessing

#### 1.1 Data Acquisition & Labeling
- ECG samples are loaded from CSV files.
- Each sample contains **188 timepoints**.
- Labels:
  - `0` → Normal  
  - `1` → Abnormal  

#### 1.2 Filtering & Denoising

Three complementary methods are implemented:

- **Wavelet Denoising (Primary)**  
  Uses Daubechies wavelets (db4) to remove high-frequency noise while preserving ECG morphology.
- **Bandpass Filtering**  
  Butterworth filter (0.05–20 Hz) removes baseline drift and high-frequency artifacts.
- **Median Filtering**  
  Eliminates impulse noise while preserving sharp transitions.

#### 1.3 Normalization

Signals are scaled to **[-1, 1]** using MinMax scaling based only on training data to avoid information leakage.

#### 1.4 Feature Extraction

Each signal is represented by **18 features**:

- R-peak statistics (4)  
- T-wave statistics (4)  
- QRS duration statistics (4)  
- RR interval statistics (4)  
- Global signal statistics (2)  
- Heart rate (1)

These features provide a compact and clinically meaningful representation.

---

### 2. Model Architectures

#### 2.1 Unsupervised – Autoencoders

Trained on **normal ECGs only**.  
Anomalies are detected via **high reconstruction error**.

- AE-1: Baseline (188 → 10 → 188)  
- AE-2: Regularized (188 → 64 → 32 → 16 → 188)  
- AE-3: Deep (188 → 128 → 64 → 32 → 188)
Threshold rule:
anomaly if reconstruction_error > mean + std

## Key Results

- Sequence models achieve **up to 99.9% accuracy**.
- CNN-LSTM provides the best accuracy-efficiency trade-off.
- Wavelet denoising significantly improves model performance.
- Feature-based ML reduces computation by **>90%** compared to raw signals.

---

## Design Rationale

- **Wavelet Filtering (db4):** Best morphology preservation  
- **18 Features:** Compact yet clinically meaningful  
- **Scaling [-1, 1]:** Improves neural convergence  
- **Stratified Split:** Prevents class imbalance bias  
- **Multiple Models:** Enables informed architectural choice  

---

## Clinical Significance

- Autoencoders handle scenarios with **no labeled anomalies**.  
- Sequence models provide **maximum accuracy** when labels exist.  
- Suitable for research, education, and real-time clinical experimentation.

---

## Technical Specifications

- Input length: 188 samples  
- Sampling rate: 360 Hz  
- Features: 18  
- Epochs: 50–80  
- Batch size: 32–512  
- Loss: Binary Crossentropy / MAE  
- Optimizer: Adam  

---

## Citation

This project was developed as part of a Deep Learning mini-project during the M1 year.

