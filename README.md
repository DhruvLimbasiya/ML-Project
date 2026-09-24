# 🫀 CardioSense AI — Cardiovascular Risk Intelligence & ML Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

**CardioSense AI** is an interactive clinical decision-support web application and machine learning platform designed to stratify cardiovascular disease (CVD) risk using patient hemodynamic profiles, biological metrics, and lifestyle parameters. Trained on a cohort of **70,000 anonymized patient records**, the platform features a calibrated Scikit-Learn machine learning pipeline integrated into a custom Streamlit UI styled with modern clinical glassmorphism aesthetics.

---

## 🌟 Key Features

* **🏠 Overview Dashboard:** High-level executive statistics, cohort demographics, disease prevalence metrics, and summary cards.
* **🔬 Patient Risk Assessment:**
  * Interactive input of patient vitals (Age, Gender, Height, Weight, Systolic/Diastolic BP, Cholesterol, Glucose, Lifestyle habits).
  * Real-time disease probability estimation powered by the trained ML pipeline (`cardiovascular_logistic_regression_pipeline.pkl`).
  * Dynamic SVG radial risk gauge indicator (Low, Moderate, High Risk).
  * Clinical blood pressure staging following **JNC-7 / ACC/AHA guidelines** (Normal, Elevated, Stage 1, Stage 2, Hypertensive Crisis).
  * Body Mass Index (BMI) auto-calculation & risk classification.
  * Individualized risk factor attribution breakdown.
* **📊 Model Analytics:** Detailed transparency into the machine learning pipeline architecture, feature scaling, model coefficients, cross-validation metrics (~71.39% accuracy), confusion matrix, and ROC-AUC curve visualizations.
* **📁 Dataset Explorer:** Searchable, filterable view of the 70,000 patient records with configurable feature range sliders and exploratory histograms.
* **🎨 Modern Clinical UI:** Styled with a custom dark-mode CSS design system (`style.css`), featuring custom cards, status pills, responsive layout grids, and clinical typography.

---

## 📊 Dataset Overview

The dataset (`cardio_train.csv`) consists of 70,000 anonymized patient records collected during clinical examinations.

| Feature | Data Type | Units / Encoding | Description |
| :--- | :--- | :--- | :--- |
| `id` | Integer | Unique identifier | Patient record ID |
| `age` | Integer | Days (converted to years in UI) | Age of patient |
| `gender` | Categorical | 1: Female, 2: Male | Patient gender |
| `height` | Integer | cm | Height in centimeters |
| `weight` | Float | kg | Weight in kilograms |
| `ap_hi` | Integer | mmHg | Systolic blood pressure |
| `ap_lo` | Integer | mmHg | Diastolic blood pressure |
| `cholesterol` | Categorical | 1: Normal, 2: Above Normal, 3: Well Above Normal | Serum cholesterol level |
| `gluc` | Categorical | 1: Normal, 2: Above Normal, 3: Well Above Normal | Fasting glucose level |
| `smoke` | Binary | 0: Non-smoker, 1: Smoker | Tobacco smoking status |
| `alco` | Binary | 0: Non-drinker, 1: Drinker | Alcohol consumption status |
| `active` | Binary | 0: Inactive, 1: Active | Physical activity status |
| **`cardio`** | **Binary Target** | **0: Absence, 1: Presence** | **Cardiovascular disease status** |

---

## ⚙️ Machine Learning Pipeline Architecture

1. **Preprocessing:** Standard scaling (`StandardScaler`) applied to numerical features (`age`, `height`, `weight`, `ap_hi`, `ap_lo`) to normalize feature distributions.
2. **Classifier:** Logistic Regression pipeline optimized with L2 regularization.
3. **Evaluation:** Evaluated using 5-fold cross-validation, achieving an accuracy of **71.39%** on test cohorts.
4. **Artifact:** Serialized pipeline object stored in `cardiovascular_logistic_regression_pipeline.pkl` for fast inference in production.

---

## 📁 Repository Structure

```
.
├── app.py                                     # Main Streamlit web application
├── style.css                                  # Custom CSS styling (Dark Glassmorphism UI)
├── cardiovascular_logistic_regression_pipeline.pkl  # Serialized ML model pipeline
├── cardio_train.csv                           # Dataset (70,000 clinical records)
├── Cardio_Task_2_fixed.ipynb                  # Jupyter Notebook: Data analysis, training & evaluation
├── Datasets.ipynb                             # Jupyter Notebook: Exploratory data analysis
├── requirements.txt                           # Python project dependencies
├── Salary_dataset.csv                         # Supplementary dataset
├── ML_SOP_Project.pdf                         # Project Statement of Purpose & Report
└── README.md                                  # Project documentation
```

---

## 🚀 Quick Start & Local Setup

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### 2. Clone Repository / Navigate to Folder
```bash
cd "your folder path"
```

### 3. Create & Activate Virtual Environment (Optional but Recommended)
* **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Application
```bash
streamlit run app.py
```

Once launched, open your web browser at `http://localhost:8501`.

---

## ⚠️ Medical Disclaimer

*This application is developed strictly for **educational, academic, and research purposes**. It is not intended for diagnostic use or to replace professional clinical judgment. Always consult a qualified healthcare professional for clinical diagnoses and treatment decisions.*

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
