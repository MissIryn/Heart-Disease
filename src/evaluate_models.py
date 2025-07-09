import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

# ==========  LOAD DATA ==========
print(" Loading data and models...")
df = pd.read_csv('../data/processed/heart_disease_cleaned.csv')
X = df.drop(columns=['Heart Disease Status'])
y = df['Heart Disease Status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==========  ENSURE OUTPUT FOLDERS EXIST ==========
eda_path = '../outputs/dataExplorationVisualization'
results_path = '../outputs/resultsAnalysisVisualization'
os.makedirs(eda_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

# ==========  EDA ==========
print("\n Running Exploratory Data Analysis (EDA)...")

# 1. Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Heart Disease Class Distribution")
plt.xlabel("Heart Disease Status")
plt.ylabel("Count")
plt.savefig(f"{eda_path}/class_distribution.png")
plt.close()

# 2. Correlation heatmap (numerical features only)
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{eda_path}/correlation_heatmap.png")
plt.close()

# 3. Violin Plot for Numerical Features by Target
print("🎻 Generating Violin Plots...")
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.violinplot(x='Heart Disease Status', y=col, data=df, palette='muted')
    plt.title(f"Violin Plot of {col} by Heart Disease Status")
    plt.tight_layout()
    plt.savefig(f"{eda_path}/violin_{col}.png")
    plt.close()


# 4. Histograms for all numerical features
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{eda_path}/hist_{col}.png")
    plt.close()

print(" EDA visualizations saved to 'outputs/dataExplorationVisualization/'.")

# ==========  LOAD MODELS ==========
print("\n Loading trained models...")
svm_model = joblib.load('../models/svm_model.pkl')
rf_model = joblib.load('../models/random_forest_model.pkl')
hybrid_model = joblib.load('../models/hybrid_svm_model.pkl')

models = {
    "SVM": svm_model,
    "Random Forest": rf_model,
    "Hybrid Stacked": hybrid_model
}

# ==========  EVALUATION ==========
plt.figure(figsize=(8, 6))
for name, model in models.items():
    print(f"\n Evaluating {name}...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = report['1']['precision'] * 100
    recall = report['1']['recall'] * 100
    f1 = report['1']['f1-score'] * 100
    auc_score = roc_auc_score(y_test, y_proba) * 100

    # Display as percentage
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1-score: {f1:.2f}%")
    print(f"AUC: {auc_score:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"{results_path}/confusion_matrix_{name.replace(' ', '_')}.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f}%)")

# Final ROC curve
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{results_path}/roc_curves_all_models.png")
plt.close()

print("\n Evaluation complete. All plots saved to 'outputs/resultsAnalysisVisualization/'.")

# ==========  SAVE METRICS REPORT TO TEXT FILE ==========
report_path = '../models/model_evaluation_report.txt'
with open(report_path, 'w') as f:
    f.write("Model Evaluation Report\n")
    f.write("========================\n")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report['1']['precision'] * 100
        recall = report['1']['recall'] * 100
        f1 = report['1']['f1-score'] * 100
        auc_score = roc_auc_score(y_test, y_proba) * 100

        f.write(f"\n{name}:\n")
        f.write(f"Precision: {precision:.2f}%\n")
        f.write(f"Recall:    {recall:.2f}%\n")
        f.write(f"F1-Score:  {f1:.2f}%\n")
        f.write(f"AUC:       {auc_score:.2f}%\n")

print(" Model evaluation metrics saved to '../models/model_evaluation_report.txt'")
