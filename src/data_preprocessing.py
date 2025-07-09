import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE
import os

# Fixed file path to go up from src/ to data/raw/
raw_data_path = '../data/raw/heart_disease.csv'

# Load raw dataset
df = pd.read_csv(raw_data_path)

# Target variable (cleaned and mapped to binary)
target = 'Heart Disease Status'
y = df[target].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
X = df.drop(columns=[target])

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Pipeline for preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ]), numerical_features),
    ('cat', Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Outlier removal
iso = IsolationForest(contamination=0.01, random_state=42)
mask = iso.fit_predict(X_processed) != -1
X_processed = X_processed[mask]
y = y[mask]

# Show class distribution before SMOTE
print(" Class distribution before SMOTE:", np.bincount(y))

# Handle class imbalance
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_processed, y)

# Show class distribution after SMOTE
print(" Class distribution after SMOTE:", np.bincount(y_balanced))

# Save processed data
processed_data = pd.DataFrame(
    X_balanced.toarray() if hasattr(X_balanced, 'toarray') else X_balanced
)
processed_data['Heart Disease Status'] = y_balanced

# Save to ../data/processed/
os.makedirs('../data/processed', exist_ok=True)
processed_data.to_csv('../data/processed/heart_disease_cleaned.csv', index=False)

print(" Preprocessing complete. Cleaned data saved to '../data/processed/heart_disease_cleaned.csv'")
