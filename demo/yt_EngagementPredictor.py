import torch
import numpy as np
from collections import deque
import time
from EngagementClassifierV1 import EngagementClassifierV1
import mediapipe as mp
from FaceFeatureExtractor import FaceFeatureExtractor, FaceFeatures
import cv2

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    static_image_mode=False,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class EngagementPredictor:
    def __init__(self, model_path: str, window_size: int = 30, threshold: float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EngagementClassifierV1().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.window_size = window_size
        self.threshold = threshold
        self.prediction_history = deque(maxlen=window_size)
        self.feature_scaler = None
        self.engagement_labels = ['Disengaged', 'Partially Engaged', 'Fully Engaged']

    def _preprocess_features(self, features: FaceFeatures) -> torch.Tensor:
        feature_vector = np.array([
            features.head_pitch,
            features.head_yaw,
            features.head_roll,
            features.gaze_x,
            features.gaze_y,
            features.eye_contact_duration,
            features.gaze_variation_x,
            features.gaze_variation_y,
            features.face_confidence,
            features.landmarks_stability,
            features.time_since_head_movement,
            features.time_since_gaze_shift
        ]).reshape(1, -1)

        if self.feature_scaler is not None:
            feature_vector = self.feature_scaler.transform(feature_vector)

        return torch.FloatTensor(feature_vector).to(self.device)

    def predict(self, features: FaceFeatures) -> dict:
        with torch.no_grad():
            input_tensor = self._preprocess_features(features)
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()

            self.prediction_history.append(prediction)

            if len(self.prediction_history) >= self.window_size:
                counts = np.bincount(list(self.prediction_history))
                smoothed_prediction = np.argmax(counts)
            else:
                smoothed_prediction = prediction

            return {
                'raw_prediction': self.engagement_labels[prediction],
                'smoothed_prediction': self.engagement_labels[smoothed_prediction],
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy().tolist()
            }

def main():
    cap = cv2.VideoCapture(0)
    feature_extractor = FaceFeatureExtractor()
    predictor = EngagementPredictor(
        model_path='best_model_v3.pth',
        window_size=30,
        threshold=0.5
    )

    blink_timestamps = deque(maxlen=100)
    saccade_buffer = deque(maxlen=30)
    fixation_durations = deque(maxlen=30)
    pupil_sizes = deque(maxlen=30)
    last_gaze = None
    last_saccade_time = time.time()
    BASELINE_PUPIL_SIZE = 20.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        overlay = np.zeros_like(frame)
        frame_h, frame_w = frame.shape[:2]
        state = "Unknown"
        fixation_avg = 0
        pupil_mean = 0
        blink_rate = 0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_idx = [33, 160, 158, 133, 153, 144]
                right_eye_idx = [362, 385, 387, 263, 373, 380]
                left_iris_idx = [468, 469, 470, 471, 472]
                right_iris_idx = [473, 474, 475, 476, 477]

                def eye_aspect_ratio(landmarks, indices):
                    p = [landmarks[i] for i in indices]
                    coords = [(int(p_.x * frame_w), int(p_.y * frame_h)) for p_ in p]
                    vert1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
                    vert2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
                    hor = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
                    return (vert1 + vert2) / (2.0 * hor)

                def pupil_diameter(landmarks, indices):
                    left = landmarks[indices[0]]
                    right = landmarks[indices[3]]
                    return np.sqrt((left.x - right.x)**2 + (left.y - right.y)**2) * frame_w

                ear_left = eye_aspect_ratio(face_landmarks.landmark, left_eye_idx)
                ear_right = eye_aspect_ratio(face_landmarks.landmark, right_eye_idx)
                ear_avg = (ear_left + ear_right) / 2
                if ear_avg < 0.20:
                    blink_timestamps.append(time.time())

                now = time.time()
                recent_blinks = [t for t in blink_timestamps if now - t < 10]
                blink_rate = len(recent_blinks) / 10

                pupil_left = pupil_diameter(face_landmarks.landmark, left_iris_idx)
                pupil_right = pupil_diameter(face_landmarks.landmark, right_iris_idx)
                pupil_avg = (pupil_left + pupil_right) / 2
                pupil_sizes.append(pupil_avg)

                current_gaze = (
                    (face_landmarks.landmark[468].x + face_landmarks.landmark[473].x) * frame_w / 2,
                    (face_landmarks.landmark[468].y + face_landmarks.landmark[473].y) * frame_h / 2
                )

                if last_gaze:
                    dx = current_gaze[0] - last_gaze[0]
                    dy = current_gaze[1] - last_gaze[1]
                    amplitude = np.sqrt(dx ** 2 + dy ** 2)
                    if amplitude > 5:
                        saccade_buffer.append({'amplitude': amplitude, 'timestamp': now})
                        fixation_durations.append(now - last_saccade_time)
                        last_saccade_time = now
                last_gaze = current_gaze

                fixation_avg = np.mean(fixation_durations) * 1000 if fixation_durations else 0
                pupil_mean = np.mean(pupil_sizes) if pupil_sizes else 0
                pupil_std_ratio = np.std(pupil_sizes) / (BASELINE_PUPIL_SIZE + 1e-5)
                distraction_score = sum([
                    blink_rate > 0.6,
                    fixation_avg < 80,
                    #pupil_std_ratio > 0.4
                ])

                features = feature_extractor.extract_features(frame, face_landmarks)
                engagement = predictor.predict(features)

                if engagement['smoothed_prediction'] == 'Fully Engaged' and distraction_score <= 1:
                    state = "Focused"
                else:
                    state = "Distracted"

                mp.solutions.drawing_utils.draw_landmarks(
                    image=overlay,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
                )

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(frame_gray, 50, 255, cv2.THRESH_BINARY_INV)
        anonymized = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        final_output = cv2.addWeighted(anonymized, 1.0, overlay, 1.0, 0)

        cv2.rectangle(final_output, (0, 0), (420, 230), (0, 0, 0), -1)
        cv2.putText(final_output, f"Blink Rate: {blink_rate:.2f}/s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(final_output, f"Fixation: {fixation_avg:.0f}ms", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(final_output, f"Pupil: {pupil_mean:.1f}px", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        state_color = (0, 255, 0) if state == "Focused" else (0, 0, 255)
        cv2.putText(final_output, state, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.8, state_color, 4)

        cv2.imshow("Simplexity_v2", final_output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
