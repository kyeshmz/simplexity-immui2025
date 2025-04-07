import joblib
from pathlib import Path
import os


class DyslexiaClassifier:
    def __init__(self, model_path=None):
        # Set default path relative to package installation
        default_model_path = Path(__file__).parent / 'models' / 'etdd70_model.pkl'

        # Allow custom model path while maintaining fallback
        self.model_path = Path(model_path) if model_path else default_model_path

        # Validate model existence
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Either:\n"
                "1. Download the pretrained model from Zenodo and place it in the models/ directory\n"
                "2. Train a new model using train_model.py"
            )

        self.model = joblib.load(self.model_path)

        # Flexible feature mapping (adjust based on your CSV columns)
        self.feature_map = {
            'avg_fixation_duration': ['fixation_duration', 'mean_fixation'],
            'num_regressions': ['regression_count', 'regressions'],
            'pupil_variability': ['pupil_diff', 'pupil_diameter_diff'],
            'saccade_ratio': ['saccade_ratio', 'saccade_amplitude_ratio']
        }

    def predict(self, features):
        """Returns confusion probability between 0-1 with feature validation"""
        expected_feature_count = len(self.feature_map)
        if len(features[0]) != expected_feature_count:
            raise ValueError(
                f"Feature dimension mismatch. Expected {expected_feature_count} features, "
                f"got {len(features[0])}. Required features: {list(self.feature_map.keys())}"
            )
        return self.model.predict_proba(features)[:, 1]

    def get_expected_features(self):
        """Return list of expected feature names with possible aliases"""
        return [f"{k} (aliases: {', '.join(v)})" for k, v in self.feature_map.items()]

# import joblib
# from pathlib import Path
#
#
# class DyslexiaClassifier:
#     def __init__(self, model_path='models/etdd70_model.pkl'):
#         self.model = joblib.load(Path(__file__).parent / model_path)
#         self.required_features = [
#             'avg_fixation_duration',
#             'num_regressions',
#             'pupil_variability',
#             'saccade_ratio'
#         ]
#
#     def predict(self, features):
#         """Returns confusion probability between 0-1"""
#         if len(features[0]) != len(self.required_features):
#             raise ValueError(f"Expected {len(self.required_features)} features, got {len(features[0])}")
#         return self.model.predict_proba(features)[:, 1]
#
#     def get_feature_names(self):
#         return self.required_features.copy()
