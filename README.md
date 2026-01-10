# ECG Anomaly Detection

A comprehensive machine learning framework for detecting cardiac anomalies in electrocardiogram (ECG) signals through advanced signal processing and deep learning methodologies.

## Abstract

This project implements an end-to-end pipeline for ECG-based cardiac anomaly detection, combining robust signal preprocessing with multiple deep learning paradigms. The approach integrates unsupervised autoencoders for reconstruction-based anomaly scoring and supervised sequence models (LSTM, Bi-LSTM, GRU, CNN-LSTM) for binary classification. The framework demonstrates high sensitivity in detecting pathological cardiac patterns while maintaining specificity on normal signals, achieving up to 99.9% accuracy with state-of-the-art architectures.

## Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Methodology](#methodology)
4. [Project Structure](#project-structure)
5. [Key Results](#key-results)



---

## Overview

Cardiac anomaly detection from ECG signals presents significant clinical and computational challenges. ECG signals are inherently noisy, subject to high inter-individual variability, and contain complex morphological patterns that require specialized analysis. This project addresses these challenges through a systematic, two-stage approach:

**Stage 1: Signal Preprocessing and Feature Engineering** – Transforms raw ECG data into clean, normalized features  
**Stage 2: Deep Learning Classification** – Employs multiple neural network architectures for anomaly detection

The framework is designed for scalability, enabling deployment in both research and clinical settings with comprehensive evaluation and interpretability.

---

## System Architecture

### End-to-End Processing Pipeline

The project follows a well-defined workflow from raw signal acquisition to clinical inference:

```
Raw ECG Signals (188 timepoints/sample)
    ↓
Data Labeling & Class Assignment
    ↓
Signal Filtering & Noise Reduction
    ↓
Normalization & Amplitude Scaling
    ↓
Cardiac Feature Extraction (18 features)
    ↓
Data Quality Verification
    ↓
Train-Test Split (80-20, Stratified)
    ↓
Model Training & Optimization
    ↓
Performance Evaluation & Validation
    ↓
Anomaly Detection & Clinical Inference
```

This systematic approach ensures data integrity, prevents information leakage, and maintains scientific rigor throughout the analysis pipeline.

![Image](./plots/complete_pipeline.png)
---

## Methodology

### 1. Signal Preprocessing

#### 1.1 Data Acquisition and Labeling

Raw ECG signals are loaded from CSV-formatted datasets with clear separation between normal and anomalous samples. Each signal represents a single heartbeat cycle or cardiac episode, consisting of 188 discrete amplitude measurements. Samples are labeled as Normal (class 0) or Anomalous (class 1), enabling supervised evaluation of model performance.

#### 1.2 Filtering and Denoising

Three complementary signal processing techniques are implemented to remove artifacts and noise while preserving clinically relevant morphological features:

**Wavelet Denoising (Primary Method)**  
Employs discrete wavelet transform decomposition using Daubechies wavelets (db4). This adaptive approach applies frequency-dependent thresholding, effectively removing high-frequency noise while preserving the essential QRS complex and wave features critical for anomaly detection. This method is selected as the default due to superior morphology preservation.

**Bandpass Filtering (Alternative)**  
Fourth-order Butterworth filter isolating the 0.05–20 Hz frequency band, effectively eliminating baseline drift (< 0.05 Hz) and high-frequency muscle noise (> 20 Hz). This approach is particularly useful for removing powerline interference and motion artifacts.

**Median Filtering (Complementary)**  
Non-linear filtering technique effective for eliminating impulse-type noise while preserving signal discontinuities and sharp morphological features. Useful as a preprocessing step before other filtering methods.

#### 1.3 Normalization and Scaling

Filtered signals are normalized to a consistent amplitude range using MinMax scaling, transforming all signals to the interval [-1, 1]. This standardization ensures numerical stability during neural network training, prevents high-amplitude signals from dominating the learning process, and improves gradient flow during backpropagation. Importantly, scaling parameters are computed from training data only to prevent information leakage to the test set.

#### 1.4 Cardiac Feature Extraction

Eighteen cardinal features are systematically extracted from preprocessed signals, capturing both morphological and rate-based characteristics essential for distinguishing normal from anomalous cardiac activity:

**R-Peak Features (4 features)**  
Identifies the dominant positive deflection in the QRS complex. Features include amplitude mean, standard deviation, median, and sum across detected R-peaks in the signal.

**T-Wave Features (4 features)**  
Captures the repolarization phase following ventricular depolarization. Extracted features represent the morphology of the T-wave through statistical summaries of detected T-peak amplitudes.

**QRS Duration Features (4 features)**  
Measures the temporal width and consistency of the ventricular depolarization phase. Statistical descriptors quantify both duration and variability across the cardiac cycle.

**RR Interval Metrics (4 features)**  
Represents inter-beat intervals, providing heart rate variability information. Essential for detecting arrhythmias and rate-related anomalies.

**Signal Statistics (2 features)**  
Overall signal mean and standard deviation, capturing baseline and amplitude characteristics.

**Derived Clinical Metric (1 feature)**  
Computed heart rate in beats per minute, calculated from detected R-peak intervals and sampling frequency.

These features form a compact (18-dimensional) yet comprehensive representation that preserves clinically meaningful information while reducing computational complexity for downstream models.

### 2. Model Architectures

#### 2.1 Unsupervised Approach: Autoencoders

Autoencoders learn compressed representations of normal ECG signals through an unsupervised mechanism. The model is trained exclusively on normal samples, learning to reconstruct them with minimal error. Anomalies are identified by exceptionally high reconstruction errors, reflecting the model's inability to efficiently compress and reconstruct abnormal patterns.

**Three Autoencoder Variants:**

**AE-1: Baseline Architecture**  
A simple autoencoder with modest compression (188 → 10 → 188). This baseline establishes fundamental reconstruction capability with minimal parameterization, useful for computational efficiency and interpretability.

**AE-2: Regularized Architecture**  
Incorporates dropout regularization with progressively increasing network capacity (188 → 64 → 32 → 16 → 188). The 16-dimensional bottleneck provides greater representational power while dropout prevents overfitting on limited normal samples.

**AE-3: Deep Architecture**  
An advanced autoencoder with maximum model capacity and aggressive regularization (188 → 128 → 64 → 32 → 188). The 32-dimensional bottleneck and elevated dropout rates (0.3) provide optimal balance between expressiveness and generalization for complex anomaly patterns.

**Key Advantages:**  
- Requires no labeled anomalous training data
- Interpretable anomaly scores based on reconstruction error
- Naturally handles class imbalance
- Anomaly threshold adaptively determined from training data distribution

**Detection Mechanism:**  
Anomalies are identified using statistical thresholding: signals with reconstruction error exceeding mean + 1 standard deviation are flagged as anomalous.

#### 2.2 Supervised Approach: Sequence Models

Sequence-based architectures leverage the temporal structure inherent in ECG signals, treating each feature sequence as a time series. These models learn discriminative patterns that characterize the boundary between normal and anomalous cardiac activity.

**LSTM (Long Short-Term Memory)**  
A recurrent architecture specifically designed to capture long-range temporal dependencies through memory cell mechanisms. Prevents vanishing gradient problems endemic to standard RNNs, enabling effective learning of temporal patterns across the entire signal duration.

**Bidirectional LSTM**  
Extends LSTM by processing information in both forward and backward temporal directions, enriching the contextual understanding of each timepoint. The bidirectional mechanism enables detection of anomalies that require both past and future context to identify.

**Stacked LSTM**  
Multiple recurrent layers arranged hierarchically, enabling learning of multi-scale temporal abstractions. Lower layers capture primitive signal patterns (e.g., individual wave morphologies), while higher layers learn semantic relationships (e.g., inter-wave relationships and rhythm patterns).

**GRU (Gated Recurrent Unit)**  
A simplified recurrent alternative to LSTM with fewer parameters (approximately 30% reduction), offering computational efficiency without significant accuracy compromise. Ideal for deployment scenarios with computational constraints.

**CNN-LSTM Hybrid**  
Combines convolutional feature extraction with recurrent temporal modeling. The convolutional layer automatically learns optimal spatial filters for ECG feature representation, followed by LSTM layers that model temporal dependencies in the extracted features. This hybrid approach achieves state-of-the-art performance through synergistic combination of spatial and temporal learning mechanisms.

**Shared Configuration:**  
All sequence models employ 64-unit recurrent layers, Adam optimization, binary crossentropy loss, and stratified train-test splits. Architecture-specific hyperparameters (e.g., dropout rates, layer dimensions) are optimized through validation-based selection.

### 3. Data Management and Validation

#### 3.1 Train-Test Splitting

Data is partitioned using stratified 80-20 split, preserving class distribution in both training and test subsets. This approach prevents artificial performance inflation from class imbalance and ensures robust generalization assessment.

#### 3.2 Data Leakage Prevention

Critical safeguards prevent information leakage between training and test sets. All preprocessing operations (filtering, scaling, feature extraction) employ parameters computed exclusively from training data. This discipline ensures that model evaluation reflects true generalization capability on unseen data.

#### 3.3 Quality Verification

Systematic verification checks ensure data integrity throughout processing: detection of missing values, identification of duplicates, verification of value ranges, and class distribution assessment. Failed verification triggers explicit alerts, preventing downstream model training on compromised data.



**File Descriptions:**

- **eda.py**: Exploratory data analysis including dataset summarization, class distribution visualization, and statistical profiling
- **filteringNoise.py**: ECG signal filtering using wavelet, bandpass, and median methods; implements class-based filtering pipeline with MSE evaluation
- **feature_extraction.py**: Extracts 18 cardiac features from filtered signals using peak detection and statistical analysis
- **visualisation.py**: Comprehensive visualization utilities including overlay plots, frequency domain analysis, statistical comparisons, and signal heatmaps
- **autoencoders.ipynb**: Implements three autoencoder variants with reconstruction-based anomaly detection
- **deepSequenceModels.py**: Creates and trains five sequence model architectures (LSTM, Bi-LSTM, Stacked LSTM, GRU, CNN-LSTM)

---

## Methodology Overview

### Phase 1: Signal Preprocessing
Raw ECG signals undergo systematic cleaning through filtering, normalization, and feature extraction. This phase ensures data quality and creates a suitable numerical representation for machine learning.

### Phase 2: Model Development
Two learning paradigms are explored:
- **Unsupervised**: Autoencoders identify anomalies through reconstruction error
- **Supervised**: Sequence models learn discriminative decision boundaries

### Phase 3: Evaluation and Comparison
Comprehensive performance evaluation compares model accuracy, precision, recall, F1-score, AUC-ROC, and computational efficiency, enabling data-driven model selection for deployment.

### Phase 4: Clinical Validation
Models are validated on held-out test data representative of clinical populations, ensuring generalization beyond training distribution.

---

## Key Results

### Model Performance Summary

The project evaluates eight distinct models across multiple performance metrics:

**Best Overall Performance: CNN-LSTM**  
- Accuracy: 97%
- Precision: 95%
- Recall: 98%
- F1-Score: 96%
- AUC-ROC: 0.99

**Performance Ranking by Accuracy:**
Sequence models achieve high accuray, precison, recall and F1-score reching 99.9%

### Key Findings

**Preprocessing Impact**  
Wavelet-based denoising provides superior morphology preservation compared to alternative filtering approaches, significantly improving downstream model performance.

**Regularization Effectiveness**  
Dropout regularization substantially reduces overfitting in deep architectures, with regularized autoencoders consistently outperforming baseline models by 2-3%.

**Architecture Complexity Trade-offs**  
Hierarchical (stacked) architectures achieve higher accuracy at the cost of increased computational overhead. The CNN-LSTM hybrid achieves optimal accuracy-efficiency balance through feature-level dimensionality reduction.

**Bidirectional Context**  
Bidirectional processing of temporal information improves discrimination by 1-2% compared to unidirectional models, reflecting the importance of contextual awareness in anomaly detection.

**Feature Engineering Value**  
Systematic feature extraction (18 features) provides competitive performance while reducing computational requirements by >90% compared to raw signal processing, making it suitable for real-time clinical deployment.

---

## Design Rationale

**Wavelet Filtering (db4, level=1)**  
Selected for optimal preservation of ECG morphology while achieving efficient noise reduction. Superior to alternatives in maintaining clinical feature integrity.

**18-Feature Representation**  
Balances clinical relevance with computational efficiency. Captures essential cardiac metrics without dimensionality explosion.

**MinMax Scaling [-1, 1]**  
Maintains signal symmetry and improves neural network convergence compared to standard 0-1 normalization. Critical for optimal performance in deep learning.

**Stratified 80-20 Split**  
Prevents class imbalance artifacts in performance evaluation. Ensures test set represents same population distribution as training data.

**Multiple Architecture Exploration**  
Systematic comparison of autoencoders and sequence models enables informed selection based on specific deployment constraints (accuracy, speed, interpretability).

**Dropout Regularization**  
Essential for preventing overfitting in architectures trained on limited anomalous samples. Substantially improves generalization capability.

---

## Clinical Significance

This framework addresses critical gaps in automated cardiac anomaly detection. The unsupervised autoencoder approach handles scenarios where anomalous training data is limited or unavailable. The supervised sequence models provide maximum accuracy when sufficient labeled data exists. The systematic comparison enables clinicians to select models balancing accuracy requirements with computational constraints of their specific deployment environment.

---

## Technical Specifications

**Signal Characteristics**  
- Input dimension: 188 timepoints per sample
- Sampling rate: 360 Hz
- Extracted features: 18 cardiac metrics

**Model Configuration**  
- Training epochs: 50-80 (architecture dependent)
- Batch size: 32-512 (optimized per architecture)
- Validation strategy: Stratified hold-out (20%)
- Loss function: Binary crossentropy (classification) / Mean absolute error (reconstruction)
- Optimizer: Adam with default parameters

---


## Citation

This project was a part of Deep Learning mini project during M1 year

```
ECG Anomaly Detection using Deep Learning and Signal Processing
Lebga Hanane, 2024
```

---

#   h a n a n e - l b g - E C G - a n o m a l i e - d e t e c t i o n  
 