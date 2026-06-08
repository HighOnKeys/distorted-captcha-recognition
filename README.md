<div align="center">

# 🔍 Distorted Visual Sequence Recognition

### A custom deep learning system that reads text from heavily distorted images

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![CER](https://img.shields.io/badge/Val_CER-0.52%25-22c55e?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-8b5cf6?style=for-the-badge)](LICENSE)

<br/>

*Built for the CIG AI/ML Challenge — no pretrained models, trained from scratch*

---

</div>

## 📌 Overview

This project solves the problem of recognizing text sequences from **heavily distorted grayscale images** — images affected by background noise, blob occlusions, blur, shape deformation, and irregular character spacing.

The solution is a **custom CRNN (Convolutional Recurrent Neural Network)** trained end-to-end using **CTC loss**, achieving a final validation Character Error Rate of **0.52%**.

<div align="center">

| Input | → | Model | → | Output |
|:---:|:---:|:---:|:---:|:---:|
| Distorted grayscale image | | Custom CRNN + CTC | | `BU522X` |

</div>

---

## 🧠 Architecture

The model follows a three-stage pipeline — visual features → sequential context → character output.

```
Input (1×64×200)
      │
      ▼
┌─────────────────────────────────────────┐
│            CNN Backbone                 │
│                                         │
│  Conv2d → BN → ReLU → MaxPool(2,2)     │  (32, 32, 100)
│  Conv2d → BN → ReLU → MaxPool(2,2)     │  (64, 16, 50)
│  Conv2d → BN → ReLU → MaxPool(2,1)     │  (128, 8, 50)
│  Conv2d → BN → ReLU → MaxPool(2,1)     │  (128, 4, 50)
└─────────────────────────────────────────┘
      │
      │  Reshape: (B, 50, 512)
      ▼
┌─────────────────────────────────────────┐
│         Bidirectional LSTM × 2          │
│                                         │
│  hidden=256, dropout=0.3, bidir=True    │  (B, 50, 512)
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│           FC + CTC Decode               │
│                                         │
│  Linear(512 → 32) + Greedy Decode       │  Text sequence
└─────────────────────────────────────────┘
```

**Total Parameters: 3,411,296 — trained entirely from scratch.**

---

## 📊 Results

<div align="center">

| Metric | Value |
|:---|:---:|
| Final Validation CER | **0.0052 (0.52%)** |
| Training Samples | 20,000 |
| Test Samples | 5,000 |
| Epochs Trained | 40 |
| Charset Size | 31 characters |

</div>

### Training Progression

| Epoch | Loss | Val CER |
|:---:|:---:|:---:|
| 1 | 3.8055 | 0.9501 |
| 10 | 3.5647 | 0.9398 |
| 20 | 0.1708 | 0.0171 |
| 30 | 0.0185 | 0.0061 |
| 40 | 0.0094 | **0.0052** |

---

## 🗂 Dataset

- **Training set:** 20,000 labeled grayscale PNG images
- **Test set:** 5,000 unlabeled images
- **Label format:** 6-character alphanumeric sequences (e.g. `BU522X`, `XQ8NE2`)
- **Charset:** `0–9`, `A–Z` (31 characters — `0`, `1` absent from training data, likely intentional to avoid visual confusion with `O` and `I`)

### Distortion Types Present

| Type | Description |
|:---|:---|
| 🔴 Background noise | Salt-and-pepper dots across the image |
| 🟠 Blob occlusion | Large black shapes obscuring characters |
| 🟡 Blur | Gaussian and motion blur artifacts |
| 🟢 Shape deformation | Characters warped and distorted |
| 🔵 Irregular spacing | Non-uniform character alignment |

### Data Cleaning

Two labels were corrupted by Excel auto-formatting during dataset creation, identified and corrected by visual inspection:

| Image | Raw Label | Corrected Label | Issue |
|:---|:---:|:---:|:---|
| `train-2184.png` | `5.40E+12` | `5396E9` | Excel scientific notation |
| `train-6819.png` | `04-Mar-54` | `4MAR54` | Excel date auto-format |

---

## ⚙️ Training Details

| Hyperparameter | Value |
|:---|:---:|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| LR Scheduler | StepLR (step=5, γ=0.5) |
| Loss Function | CTC Loss (blank=0) |
| Batch Size | 64 |
| Epochs | 40 |
| Gradient Clipping | max norm 5 |
| Train/Val Split | 90% / 10% |
| Image Size | 64 × 200 (H × W) |

---

## 🚀 Reproduce

> **Requirements:** Google Colab with GPU runtime (T4 or better recommended)

**Step 1** — Open `notebook.ipynb` in Google Colab

**Step 2** — Set runtime to GPU: `Runtime → Change runtime type → T4 GPU`

**Step 3** — Run Cell 1 (Imports). When Cell 3 executes, upload the dataset zip when prompted.

**Step 4** — Run all remaining cells in order. Training takes ~20–25 minutes on T4.

**Step 5** — Cell 11 auto-downloads `submission.csv` on completion.

---

## 📁 Repository Structure

```
distorted-captcha-recognition/
│
├── notebook_Kumar_Manas_23119016.ipynb          # Complete solution — all cells with outputs
├── submission_Kumar_Manas_23119016.csv          # Final predictions on 5000 test images
├── assets/
│   └── sample_images.png   # Visualization of training samples
└── README.md
```

---

## 🔑 Key Design Decisions

**Why CRNN?**
CRNNs are purpose-built for sequence recognition from images. The CNN extracts spatial features column by column; the BiLSTM reads those columns as a sequence, capturing left-right character context. CTC loss handles the alignment problem — we never need to know where each character starts and ends in the image.

**Why CTC over cross-entropy?**
Cross-entropy requires knowing the exact position of each character in the image. CTC does not — it learns to align predictions to labels automatically. This is critical for distorted images where character positions are irregular.

**Why greedy decode over beam search?**
At 0.52% CER, greedy decoding is sufficient. Beam search adds inference latency for marginal gains at this error rate.

---

## 📋 Submission Format

```csv
image,prediction
test-0.png,QVTQ8A
test-1.png,7PSW9D
test-2.png,WJ2WNY
```

---

## Author

**Kumar Manas**
B.Tech. Production and Industrial Engineering · IIT Roorkee

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kumarmanas-iitroorkee/)
[![GitHub](https://img.shields.io/badge/GitHub-121011?style=flat&logo=github&logoColor=white)](https://github.com/HighOnKeys)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:kumar_m@me.iitr.ac.in)

---

<div align="center">

Built with PyTorch · Trained on Google Colab · CIG AI/ML Challenge 2026

</div>
