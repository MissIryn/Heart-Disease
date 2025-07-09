# Heart Disease Prediction Using a Hybrid Machine Learning Approach

This project applies a hybrid machine learning model—**Random Forest + Support Vector Machine (SVM)**—to improve the accuracy of heart disease prediction using patient health records.

---

##  Project Structure

```
PROJECT/
├── data/
│   ├── raw/                          # Original dataset (CSV format)
│   └── processed/                    # Cleaned dataset after preprocessing
│
├── models/                           # Trained models (.pkl) and accuracy log
│   ├── svm_model.pkl
│   ├── random_forest_model.pkl
│   ├── hybrid_svm_model.pkl
│   └── model_accuracy_log.txt
│
├── outputs/
│   ├── dataExplorationVisualization/      # EDA plots (distributions, heatmaps)
│   └── resultsAnalysisVisualization/      # Evaluation results (ROC, confusion)
│
├── src/                             # Python scripts
│   ├── data_preprocessing.py
│   ├── train_models.py
│   └── evaluate_models.py
│
├── requirements.txt
└── README.md
```

---

##  Technologies Used

- Python 3.13+
- scikit-learn
- imbalanced-learn (`SMOTE`)
- pandas
- numpy
- matplotlib
- seaborn
- joblib

---

##  Installation

1. Clone the repository or copy the project files.
2. Set up a Python virtual environment (recommended).
3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  Data Preprocessing

Run the preprocessing script to:
- Impute missing values
- Normalize numerical features
- Encode categorical features
- Remove outliers
- Apply SMOTE for class balance

```bash
cd src
python data_preprocessing.py
```

Output saved to:
```
data/processed/heart_disease_cleaned.csv
```

---

##  Model Training

Trains and saves:
- Optimized SVM
- Random Forest
- Hybrid Stacked Model (RF + SVM with Logistic Regression)

```bash
python train_models.py
```

Models saved in `models/`, accuracy logged in `model_accuracy_log.txt`.

---

##  Evaluation & Visualization

Generates:
- Precision, Recall, F1-score, AUC
- Confusion Matrices
- ROC Curves
- EDA Charts

```bash
python evaluate_models.py
```

Visuals saved in:
- `outputs/dataExplorationVisualization/`
- `outputs/resultsAnalysisVisualization/`

---

##  Results Summary

| Model            | Precision | Recall | F1-score | AUC    |
|------------------|-----------|--------|----------|--------|
| SVM              | 84.67%    | 88.19% | 86.39%   | 93.46% |
| Random Forest    | 99.83%    | 75.95% | 86.27%   | 93.02% |
| Hybrid Stacked   | 95.87%    |83.52%  | 89.27%   | 94.65% |

---

##  Objective Achieved

The hybrid model successfully outperformed individual models, meeting the goal of achieving  **high prediction accuracy for heart disease detection**.

---

##  Author

Irene Arthur  
BSc Computer Science Final Year  
2025
