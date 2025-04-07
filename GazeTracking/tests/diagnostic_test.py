# tests/diagnostic_test.py
from dataset import ETDD70Loader
from analysis import DyslexiaClassifier
import joblib


def run_validation():
    # Load sample data
    loader = ETDD70Loader()
    df = loader.load_task_data(['Pseudo_Text'])

    # Load model
    classifier = DyslexiaClassifier()

    # Test feature compatibility
    sample_features = df.iloc[0][classifier.FEATURE_COLS].values.reshape(1, -1)

    # Should return probability between 0-1
    prediction = classifier.predict(sample_features)[0]
    print(f"Validation result: {prediction:.2f} (Expected: 0.3-0.7 range)")


if __name__ == "__main__":
    run_validation()
