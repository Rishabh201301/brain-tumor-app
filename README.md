# Detection of Brain Tumor from MRI Images
### Mini Project • 5th Semester • Course Code: BCS554
**ABES Engineering College, Ghaziabad**  
**Department of Computer Science & Engineering**  
**Academic Session: 2026-27**

---

## 📌 Project Overview

This project is a clinical decision-support web application for the screening, segmentation, classification, and explainable visualization of brain tumors from Magnetic Resonance Imaging (MRI) scans. Built using **Streamlit**, **OpenCV**, and **NumPy**, it provides an intuitive, responsive medical dashboard designed for academic demonstration and research prototyping.

> **Clinical Disclaimer:**  
> This system is designed as an educational and decision-support aid only. It does not replace clinical judgment, and the final diagnosis rests strictly with a qualified medical specialist.

---

## 👥 Academic & Project Details

- **Project ID:** `PJ_ID_2026_CSE_62`
- **Supervisor:** Mr. Vivek Chaudhary
- **Team Members:**
  - **Ranjan Yadav** (Roll No: `2400320100889`)
  - **Rishabh Dubey** (Roll No: `2400320100900`)
  - **Rishabh Gupta** (Roll No: `2400320100901`)

---

## ⚙️ Diagnostic Pipeline Architecture

```
[Input MRI Scan]
       │
       ▼
[Preprocessing] ─── Standardize to 256x256, Gaussian Denoising & Normalization
       │
       ▼
[Binary Screening] ─── Screen for Tumor Presence (Positive vs Negative)
       │
       ├─► [No Tumor Detected] ─── Display clearance and halt pipeline
       │
       ▼ (Tumor Detected)
[Segmentation] ─── Isolate Tumor Region via Morphological Filtering & Otsu Threshold
       │
       ├─► [Feature Extraction] ─── Radiomic & Statistical Metrics (Mean, Std, Coverage)
       │
       ├─► [Surface Area] ─── Calculate Cross-Sectional Area (mm²) from In-Plane Pixel Spacing
       │
       ├─► [Classification] ─── Subtype Prediction (Glioma, Meningioma, Pituitary)
       │
       ├─► [Explainability] ─── Grad-CAM Activation Heatmap (JET Colormap Overlay)
       │
       ▼
[Clinical Summary Report] ─── Export High-Resolution PNG Diagnostic Sheet
```

---

## 📁 Repository Structure

```
brain-tumor-app/
├── .streamlit/
│   └── config.toml          # Academic theme palette (#1F2A44, #FBF3EA, Serif)
├── assets/
│   ├── abes_logo.png        # Institution logo placeholder
│   └── sample_mri.png       # Sample test MRI scan for verification
├── models/
│   └── efficientnetb3.onnx  # Pre-trained 4-class Deep Learning model (ONNX Runtime)
├── .venv/                   # Python virtual environment (ignored in git)
├── app.py                   # Streamlit interactive medical dashboard UI
├── pipeline.py              # Backend inference, segmentation, classification, and Grad-CAM
├── requirements.txt         # Core dependencies
└── README.md                # Comprehensive documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10 or higher (Tested with Python 3.13)
- `pip` package manager

### 2. Clone or Navigate to the Project Directory
```bash
cd "brain-tumor-app"
```

### 3. Create & Activate a Virtual Environment
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Streamlit Application
```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

---

## 🔬 Key Features

1. **Step-by-Step Pipeline Tracker**:
   - Visual progress indicator marking: `Input MRI -> Preprocess -> Segment -> Extract Features -> Measure Area -> Classify` with real-time completion checkmarks.
2. **Interactive Visualization Tabs**:
   - **Binary Segmentation**: Clean mask isolating the tumor mass from background parenchyma.
   - **Tumor Overlay**: Transparent crimson overlay with sharp boundary delineation over the original scan.
   - **Grad-CAM Saliency**: Focused JET heatmap showing regions influential to the model's prediction.
3. **Quantitative Metrics**:
   - Subtype classification with confidence percentages.
   - Cross-sectional surface area estimation ($\text{mm}^2$) calibrated against user-defined pixel spacing.
4. **Clinical Export**:
   - **Download Report**: Generates a unified, high-resolution PNG diagnostic summary complete with patient scan comparisons and institutional credentials.
5. **Responsive Laptop Layout**:
   - Medical dashboard styling with rounded cards, subtle shadows, and zero horizontal scrolling on standard laptop displays.
