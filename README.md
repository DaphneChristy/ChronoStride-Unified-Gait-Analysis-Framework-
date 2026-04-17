# AI Gait Recognition System

*A Unified Deep Learning Framework for Healthcare and Forensic Applications*

## Introduction

Human gait is a unique behavioral biometric that represents how a person walks. Unlike fingerprints or facial recognition, gait can be captured **from a distance without cooperation**, making it ideal for surveillance and healthcare monitoring. 
This project proposes a **unified AI-based gait recognition system** that analyzes walking patterns using deep learning to:

* Detect **neurological disorders**
* Perform **forensic suspect identification**
* Provide **automated analysis and reports**

---

## Objectives

* Build a system for **accurate gait-based identification**
* Detect diseases like **Parkinson’s and Alzheimer’s**
* Combine **healthcare + forensic modules** in one framework
* Improve performance using **advanced deep learning models**
* Handle real-world challenges like **occlusion, noise, and variation** 

---

## Key Features

* Person identification using gait
* Disease prediction from walking patterns
* Video-based gait analysis
* Feature visualization and comparison graphs
* Automated report generation
* Real-time analysis (Streamlit-based UI)

---

## Technologies Used

### Core Domains

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Computer Vision

### Models & Techniques

* Convolutional Neural Networks (CNN)
* Long Short-Term Memory (LSTM)
* Hybrid CNN-LSTM Model
* Graph Transformer Networks
* Self-Supervised Learning
* Variational Autoencoders (VAE)
* Diffusion Models 

---

## System Architecture

### Workflow

1. **Data Acquisition**

   * Healthcare: Sensor data
   * Forensic: CCTV/video input

2. **Preprocessing**

   * Noise removal
   * Background subtraction
   * Silhouette extraction

3. **Feature Extraction**

   * Spatial features (body shape, posture)
   * Temporal features (movement patterns)

4. **Model Processing**

   * CNN → spatial learning
   * LSTM → temporal learning
   * Graph Transformer → joint relationships

5. **Output**

   * Disease prediction (Healthcare)
   * Suspect identification (Forensic) 

---

## Healthcare Module

* Detects neurological disorders using gait signals
* Uses statistical features:

  * Mean
  * Standard deviation
  * Max / Min
  * Median

### Diseases Detected

* Parkinson’s Disease
* Alzheimer’s Disease
* Normal Gait

### Output

* Prediction label
* Confidence score
* Clinical interpretation
* Medical suggestions 

---

## Forensic Module

* Identifies individuals using walking patterns
* Works on CCTV footage

### Features Extracted

* Step length
* Stride length
* Walking speed
* Cadence

### Output

* Suspect ID
* Match confidence
* Investigation status 

---

##  Datasets Used

### Healthcare Dataset

* PhysioNet Gait Dataset
* Includes Parkinson’s, Alzheimer’s, and Normal samples

### Forensic Dataset

* CASIA-B Dataset
* Multiple viewing angles and walking conditions 

---

## Results

### Healthcare

* Accuracy: **86.7%**
* Significant improvement over traditional ML models

### Forensic

* Accuracy: **91.2% (Graph Transformer)**
* Better performance than KNN, SVM, CNN-LSTM

These results prove deep learning models outperform traditional methods in gait analysis. 

---

## Implementation

* Built using **Python + Streamlit**
* Includes:

  * Healthcare dashboard
  * Forensic dashboard
  * Video analysis module
  * Dataset visualization
  * Model performance tracking

---

## Future Enhancements

* Real-time surveillance integration
* Support for more diseases (ALS, stroke)
* 3D gait analysis using LiDAR
* Federated learning for privacy
* Lightweight models for edge devices 

---

## References

Includes IEEE, CVPR, AAAI, and Nature Digital Medicine papers related to:

* Gait recognition
* Deep learning models
* Healthcare AI systems
* Forensic analysis 

