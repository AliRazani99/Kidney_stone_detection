# Kidney Stone Detection Challenge

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)  
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](#requirements)  
[![PyTorch](https://img.shields.io/badge/pytorch-1.10%2B-orange.svg)](#requirements)

A deep learning project for **detecting** and **localizing** kidney stones in ultrasound images, using image classification and object detection techniques.

---


---

## 🔍 Overview

Kidney stones are a common medical condition that can be diagnosed using ultrasound imaging. This challenge walks you through:

1. **Image Classification** — build a CNN-based classifier to distinguish “Normal” vs. “Stone” images.  
2. **Object Detection** — manually annotate stone regions, then train a detector (e.g. YOLOv8 or Faster R-CNN) to localize stones automatically.

---

## 📂 Dataset

- **Total images:** 9,416 ultrasound scans  
  - 4,414 Normal  
  - 5,002 Stone  
- **Resolution:** 512 × 512 pixels  
- **Sources:** Multiple hospitals/scan centers (SAMSUNG RS85, HS60, RS80A, HS70A)  
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---