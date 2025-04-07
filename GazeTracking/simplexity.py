"""
Simplexity: Real-time Confusion Detection with Gaze Tracking
ETDD70 Citation:
Dostalová, N., et al. (2024). Eye-tracking Dyslexia Dataset.
Zenodo. https://doi.org/10.5281/zenodo.13332134
"""

import cv2
from gaze_tracking import GazeTracking
from gaze_tracking.dataset import ETDD70Loader
from gaze_tracking.analysis import DyslexiaClassifier


class ConfusionAwareGaze(GazeTracking):
    def __init__(self):
        super().__init__()
        self.fixation_history = []
        self.saccade_history = []
        self.last_gaze = None

    def get_etdd70_features(self):
        """Extract ETDD70-compatible features in real-time"""
        features = {
            'avg_fixation_duration': self._calculate_avg_fixation(),
            'num_regressions': self._count_regressions(),
            'pupil_variability': abs(self.pupil_left_diameter - self.pupil_right_diameter),
            'saccade_ratio': self._calculate_saccade_ratio()
        }
        return [list(features.values())]

    def _calculate_avg_fixation(self):
        return sum(self.fixation_history) / len(self.fixation_history) if self.fixation_history else 0

    def _count_regressions(self):
        return len([s for s in self.saccade_history if s.get('direction') == 'left'])

    def _calculate_saccade_ratio(self):
        if len(self.saccade_history) < 2:
            return 0
        return len([s for s in self.saccade_history if s.get('amplitude', 0) > 30]) / len(self.saccade_history)


if __name__ == "__main__":
    gaze = ConfusionAwareGaze()
    classifier = DyslexiaClassifier()
    webcam = cv2.VideoCapture(0)

    print("📸 Starting Simplexity. Press ESC to exit.")

    while True:
        _, frame = webcam.read()
        gaze.refresh(frame)

        try:
            features = gaze.get_etdd70_features()
            confusion_prob = classifier.predict(features)[0]
            status = f"Confusion risk: {confusion_prob:.0%}"
            color = (0, 255, 0) if confusion_prob < 0.5 else (0, 0, 255)
            cv2.putText(frame, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Simplexity v2", frame)
        except Exception as e:
            print(f"⚠️ Error: {str(e)}")

        if cv2.waitKey(1) == 27:  # ESC
            break

    webcam.release()
    cv2.destroyAllWindows()
