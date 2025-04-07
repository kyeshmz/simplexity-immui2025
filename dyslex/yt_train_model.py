# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

# === Config ===
features_csv = "data/experiment-final/v3.0/extracted_features/T4_Comprehension_features.csv"
model_out_path = "models/mlp_t4.joblib"
scaler_out_path = "models/scaler_t4.joblib"

# === Load feature data ===
df = pd.read_csv(features_csv, sep=';')
print("✅ Loaded feature file:", features_csv)

# Drop subject_id if present
X = df.drop(columns=['subject_id', 'label'], errors='ignore')
y = df['label']

# === Split and scale ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train MLP ===
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
mlp.fit(X_train_scaled, y_train)

# === Evaluate ===
y_pred = mlp.predict(X_test_scaled)
print("🧪 Evaluation:")
print(classification_report(y_test, y_pred))

# === Save model + scaler ===
os.makedirs("models", exist_ok=True)
joblib.dump(mlp, model_out_path)
joblib.dump(scaler, scaler_out_path)
print(f"💾 Saved MLP model to: {model_out_path}")
print(f"💾 Saved scaler to: {scaler_out_path}")
