import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print(" Loading preprocessed data...")
processed_path = '../data/processed/heart_disease_cleaned.csv'
df = pd.read_csv(processed_path)

X = df.drop(columns=['Heart Disease Status'])
y = df['Heart Disease Status']

print(" Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ==== 1. Optimized SVM ====
print(" Optimizing Support Vector Machine (SVM) with GridSearchCV...")

svm_params = {
    'C': [1, 10, 100, 1000],
    'gamma': ['scale', 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}

svm_grid = GridSearchCV(
    SVC(probability=True),
    param_grid=svm_params,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)

svm_best = svm_grid.best_estimator_
svm_train_acc = accuracy_score(y_train, svm_best.predict(X_train)) * 100
svm_test_acc = accuracy_score(y_test, svm_best.predict(X_test)) * 100
print(f" Optimized SVM trained. Train Accuracy: {svm_train_acc:.2f}% | Test Accuracy: {svm_test_acc:.2f}%")

# ==== 2. Train Random Forest (RF) ====
print(" Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf.fit(X_train, y_train)
rf_train_acc = accuracy_score(y_train, rf.predict(X_train)) * 100
rf_test_acc = accuracy_score(y_test, rf.predict(X_test)) * 100
print(f" Random Forest trained. Train Accuracy: {rf_train_acc:.2f}% | Test Accuracy: {rf_test_acc:.2f}%")

# ==== 3. Hybrid Model Using Stacking ====
print(" Building Hybrid Model with StackingClassifier (RF + SVM)...")

# Base learners
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True))
]

# Meta-learner
meta_learner = LogisticRegression()

# Stacked classifier
hybrid_stacked = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1,
    passthrough=False
)

# Train stacked model
hybrid_stacked.fit(X_train, y_train)

# Evaluate stacked model
hybrid_train_acc = accuracy_score(y_train, hybrid_stacked.predict(X_train)) * 100
hybrid_test_acc = accuracy_score(y_test, hybrid_stacked.predict(X_test)) * 100
print(f" Hybrid Stacked Model trained. Train Accuracy: {hybrid_train_acc:.2f}% | Test Accuracy: {hybrid_test_acc:.2f}%")

# ==== Save models ====
print(" Saving models...")
os.makedirs('../models', exist_ok=True)
joblib.dump(svm_best, '../models/svm_model.pkl')
joblib.dump(rf, '../models/random_forest_model.pkl')
joblib.dump(hybrid_stacked, '../models/hybrid_svm_model.pkl')
 
 # ==== Prepare Data for Accuracy Visualization ====
models = ["SVM", "Random Forest", "Hybrid Stacked"]

train_acc = [svm_train_acc, rf_train_acc, hybrid_train_acc]
test_acc = [svm_test_acc, rf_test_acc, hybrid_test_acc]

# ==== Grouped Bar Chart ====
x = np.arange(len(models))  # model indices
width = 0.35  # bar width

fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x - width/2, train_acc, width, label='Train Accuracy')
bars2 = ax.bar(x + width/2, test_acc, width, label='Test Accuracy')

# Labels and formatting
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Models')
ax.set_title('Model Accuracy Performance (Train vs Test)')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 110)

# Annotate bars with values
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

# Save visualization
plt.tight_layout()
plt.savefig("../outputs/resultsAnalysisVisualization/model_accuracy_only.png")
plt.close()

print("✅ Accuracy visualization (Train vs Test) saved to '../outputs/resultsAnalysisVisualization/model_accuracy_only.png'")

# ==== Log model performance ==== 
with open("../models/model_accuracy_log.txt", "w") as f:
    f.write("Model Performance Summary\n")
    f.write("==========================\n")
    f.write("Dataset Split: 80% Training | 20% Testing\n\n")
    f.write(f"SVM         - Train Accuracy: {svm_train_acc:.2f}% | Test Accuracy: {svm_test_acc:.2f}%\n")
    f.write(f"RandomForest - Train Accuracy: {rf_train_acc:.2f}% | Test Accuracy: {rf_test_acc:.2f}%\n")
    f.write(f"Hybrid      - Train Accuracy: {hybrid_train_acc:.2f}% | Test Accuracy: {hybrid_test_acc:.2f}%\n")

print(" Accuracy results saved to '../models/model_accuracy_log.txt'")
print(" All models saved and training completed.")
