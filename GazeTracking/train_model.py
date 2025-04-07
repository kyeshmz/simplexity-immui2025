import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# === CONFIG ===
FEATURE_FILE = "v3.0/extracted_features/T3_meaningful_text_features.csv"
MODEL_DIR = "models"
MODEL_NAME = "mlp_t3.joblib"
SCALER_NAME = "scaler_t3.joblib"
LABEL_COLUMN = "label"  # Change if needed!

# === Load dataset ===
print(f"📄 Loading features from {FEATURE_FILE}")
df = pd.read_csv(FEATURE_FILE, sep=';')

# === Prepare data ===
if LABEL_COLUMN not in df.columns:
    raise ValueError(f"Label column '{LABEL_COLUMN}' not found in CSV")

X = df.drop(columns=["subject_id", LABEL_COLUMN], errors='ignore')  # Keep only numeric features
y = df[LABEL_COLUMN]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Scale ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train MLP ===
print("🧠 Training MLP model...")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)

# === Evaluate ===
y_pred = mlp.predict(X_test_scaled)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === Save model and scaler ===
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(mlp, os.path.join(MODEL_DIR, MODEL_NAME))
joblib.dump(scaler, os.path.join(MODEL_DIR, SCALER_NAME))

print(f"📦 Model saved to: {os.path.join(MODEL_DIR, MODEL_NAME)}")
print(f"📦 Scaler saved to: {os.path.join(MODEL_DIR, SCALER_NAME)}")



# # train_model.py
# from dataset import ETDD70Loader
# from sklearn.ensemble import RandomForestClassifier
# import joblib
#
# loader = ETDD70Loader()
# df = loader.load_task_data()
#
# X = df[['avg_fixation_duration', 'num_regressions',
#         'pupil_variability', 'saccade_ratio']]
# y = df['label']
#
# model = RandomForestClassifier(n_estimators=100)
# model.fit(X, y)
#
# joblib.dump(model, 'gaze_tracking/models/etdd70_model.pkl')
