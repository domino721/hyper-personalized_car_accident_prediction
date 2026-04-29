# 🚗 Hyper-Personalized Car Accident Prediction

> **자동차 보험 고객 데이터 기반 사고 유무 예측 초개인화 모델링**

A machine learning pipeline that predicts individual car accident probability using auto-insurance customer data, enabling hyper-personalized risk assessment.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Models](#models)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Results](#results)
- [License](#license)

---

## Overview

This project builds a **hyper-personalized car accident prediction model** by leveraging auto-insurance customer profiles. It explores multiple ML/DL approaches—from classical models (Logistic Regression, KNN, SVM, Random Forest, XGBoost, LightGBM) to deep learning architectures (LSTM, BERT)—and combines them via ensemble methods to maximize predictive accuracy.

### Key Features

| Feature | Description |
| --- | --- |
| **Multi-Model Comparison** | Systematic evaluation across 8+ algorithms |
| **Feature Engineering** | Derived variables from accident rate time-series and customer demographics |
| **LSTM Time-Series** | Age/gender-segmented accident rate prediction using LSTM |
| **BERT NLP** | Text-based feature extraction via custom BERT tokenization |
| **Ensemble** | Final prediction combining best-performing models |

---

## Project Structure

```
hyper-personalized_car_accident_prediction/
│
├── src/                            # Modular source code
│   ├── config.py                  # Global configurations & mappings
│   ├── data_processor.py          # Data cleaning & preprocessing logic
│   └── model_trainer.py           # Model training & evaluation logic
│
├── tests/                          # Automated unit tests
│   └── test_data_processor.py     # Tests for data processing logic
│
├── main.py                         # Professional pipeline entry point
│
├── Dataset/                        # Raw & processed datasets
│   ├── 사고율 LSTM 학습용 Dataset/ # Accident rate data for LSTM training
│   └── 파생변수 Dataset/          # Derived feature datasets
│
├── EDA/                            # Exploratory Data Analysis notebooks
│   ├── EDA_EA.ipynb
│   ├── EDA_MH.ipynb
│   ├── EDA_R.ipynb
│   └── EDA_WJ.ipynb
│
├── Model/                          # Model implementations (Notebooks)
│   ├── Preprocess/                 # Data preprocessing pipelines
│   │   └── Labeling/              # Target labeling logic
│   ├── LR/                        # Logistic Regression
│   ├── KNN/                       # K-Nearest Neighbors
│   ├── SVM/                       # Support Vector Machine
│   ├── RF/                        # Random Forest
│   ├── XGB/                       # XGBoost
│   ├── LSTM/                      # LSTM time-series model
│   └── BERT/                      # BERT-based NLP feature model
│
├── Ensemble/                       # Ensemble model for final prediction
│
├── Results/                        # Output & evaluation results
│   ├── rf_roc_curve.png           # Example modular pipeline output
│   └── rf_confusion_matrix.png    # Example modular pipeline output
│
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

---

## Models

### Classical ML Models

| Model | Description | Notebook(s) |
| --- | --- | --- |
| **Logistic Regression** | Baseline + multi-round hyperparameter tuning with derived features | `Model/LR/` |
| **K-Nearest Neighbors** | Distance-based classification baseline | `Model/KNN/` |
| **Support Vector Machine** | Kernel-based classification with GradBoost comparison | `Model/SVM/` |
| **Random Forest** | Tree-ensemble baseline | `Model/RF/` |
| **XGBoost** | Gradient boosting with full tuning pipeline | `Model/XGB/` |
| **LightGBM** | Gradient boosting (results in `Results/lgbm.csv`) | — |

### Deep Learning Models

| Model | Description | Notebook(s) |
| --- | --- | --- |
| **LSTM** | Time-series accident rate prediction segmented by age/gender | `Model/LSTM/` |
| **BERT** | Custom WordPiece tokenization for text-feature extraction | `Model/BERT/` |

### Ensemble

Combined prediction using the best-performing models (`Ensemble/`).

---

## Dataset

> ⚠️ **Note**: Dataset files are excluded from this repository due to size constraints (total \~170 MB). See instructions below.

### Source Data

The dataset originates from auto-insurance customer records and contains:

- **Customer Demographics**: Age group, gender, insurance period
- **Vehicle Information**: Domestic/foreign, vehicle type, effective vehicle count
- **Insurance Details**: Driver limitation coverage, special agreements
- **Accident History**: 3-year accident count, accident occurrence (target variable)

### File Descriptions

| File | Description | Size |
| --- | --- | --- |
| `(자동차보험) 고객별 사고 발생률 예측 모델링_1.csv` | Primary customer dataset | \~95 MB |
| `(자동차보험) 고객별 사고 발생률 예측 모델링_2.csv` | Supplementary dataset #2 | \~8 MB |
| `(자동차보험) 고객별 사고 발생률 예측 모델링_3.csv` | Supplementary dataset #3 | \~4 MB |
| `(자동차보험) 고객별 사고 유무 기본 데이터.csv` | Base accident occurrence data | \~25 MB |
| `base_process.csv` | Preprocessed base dataset | \~17 MB |

### How to Obtain the Data

1. Place the dataset files in the `Dataset/` directory
2. Ensure the file names match those listed above
3. Run the preprocessing notebooks in `Model/Preprocess/` to generate derived features

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/hyper-personalized_car_accident_prediction.git
cd hyper-personalized_car_accident_prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 1. Modular Pipeline (Recommended)

For a professional and automated execution of the preprocessing and model training pipeline:

```bash
# Run the complete pipeline
py main.py
```

This will automatically load the data, clean it using `DataProcessor`, and train/evaluate a model using `ModelTrainer`. Results and plots will be saved in the `Results/` directory.

### 2. Manual Workflow (Notebooks)

If you prefer exploring the logic step-by-step:

1. **EDA**: Run notebooks in `EDA/` to understand data distributions.
2. **Preprocess**: Run `Model/Preprocess/` notebooks for feature engineering.
3. **Train**: Train individual models in `Model/<MODEL_NAME>/`.
4. **Ensemble**: Combine predictions in `Ensemble/`.
5. **Evaluate**: Check results in `Results/`.

### 3. Running Tests

To ensure the modular components are working correctly:

```bash
# Run unit tests
py tests/test_data_processor.py
```

---

## Results

Model evaluation results are stored in `Results/`:

- `Results/LR/LR_total_result.csv` — Logistic Regression comprehensive results
- `Results/lgbm.csv` — LightGBM prediction output

Detailed performance metrics and visualizations are available within each model's notebook.

---

## Tech Stack

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" /><img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter" /><img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn" /><img src="https://img.shields.io/badge/XGBoost-Boosting-006400?style=flat-square" alt="XGBoost" /><img src="https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow" /><img src="https://img.shields.io/badge/PyTorch-BERT-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />

---

## License

This project is for academic and research purposes. Please contact the authors for commercial use.

---

본 프로젝트는 자동차보험 고객 데이터를 기반으로 **개인별 사고 발생 확률을 예측**하는 초개인화 모델링 파이프라인입니다.

- **데이터**: 고객 인구통계, 차량 정보, 보험 특약, 사고 이력 등
- **모델**: LR, KNN, SVM, RF, XGBoost, LightGBM, LSTM, BERT
- **파이프라인**: EDA → 전처리 → 모델 학습 → 앙상블 → 평가
- **목표**: 다양한 ML/DL 모델의 성능을 비교하고, 앙상블을 통해 최적의 예측 모델을 구축