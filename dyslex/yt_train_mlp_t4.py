import os
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from feature_extractor import load_and_transform_subject_characteristics, create_subject_characteristics_profile


# === Paths ===
DATA_FILE = "data/experiment-final/v3.0/extracted_features/T4_Comprehension_features.csv"
LABEL_FILE = "meta/subject_class_mapping-v3.0.txt"
OUTPUT_DIR = "trained_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Data ===
print("📥 Loading features:", DATA_FILE)
df = pd.read_csv(DATA_FILE, sep=';')

print("📥 Loading labels:", LABEL_FILE)
labels_df = pd.read_csv(LABEL_FILE)
df = df.merge(labels_df[['subject_id', 'class_id']], on='subject_id', how='left')

X = df.drop(columns=["subject_id", "class_id"]).astype(np.float32)
y = df["class_id"].astype(int)

# === Preprocessing ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# === Train MLP ===
print("🧠 Training MLP...")
clf = MLPClassifier(hidden_layer_sizes=(X.shape[1] // 2,), max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === Save ===
joblib.dump(clf, os.path.join(OUTPUT_DIR, "mlp_t4.joblib"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler_t4.joblib"))
print("💾 Saved model and scaler to:", OUTPUT_DIR)
