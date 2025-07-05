# train_model.py

import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import joblib

# === Paths ===
DATA_PATH = os.path.join("data", "Training.csv")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load and clean dataset ===
df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.dropna(inplace=True)

# === Features and target ===
target_col = 'prognosis'
X = df.drop(columns=[target_col])
y = df[target_col]

# === Encode target ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === Train model ===
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# === Evaluate model ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print(f"✅ Model Trained — Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

# === Save model and metadata ===
joblib.dump(model, os.path.join(MODEL_DIR, "rf_disease_model.pkl"))
joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
joblib.dump([col.replace('_', ' ').lower() for col in X.columns],  # preprocessed for NLP matching
            os.path.join(MODEL_DIR, "feature_names.pkl"))

print("📦 Model, encoder, and feature names saved successfully.")
