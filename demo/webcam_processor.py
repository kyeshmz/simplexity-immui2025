import torch
import numpy as np
from collections import deque
import time
import cv2
import mediapipe as mp
from PySide6.QtCore import QObject, Signal

# Import the existing classes
from EngagementClassifierV1 import EngagementClassifierV1
from FaceFeatureExtractor import FaceFeatureExtractor, FaceFeatures

class WebcamProcessor(QObject):
    frame_processed = Signal(np.ndarray, dict)  # Signal emitting processed frame and data
    concentration_updated = Signal(float)  # Signal to update concentration

    def __init__(self, model_path="best_model_v3.pth"):
        super().__init__()
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            static_image_mode=False,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize engagement predictor components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Try to load the model
            self.model = EngagementClassifierV1().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.model_loaded = True
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            # Fall back to simpler engagement detection if model can't be loaded
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Using fallback engagement detection method")
            self.model = None
            self.model_loaded = False
        
        # Initialize feature extractor
        self.feature_extractor = FaceFeatureExtractor()
        
        # Tracking variables
        self.blink_timestamps = deque(maxlen=100)
        self.saccade_buffer = deque(maxlen=30)
        self.fixation_durations = deque(maxlen=30)
        self.pupil_sizes = deque(maxlen=30)
        self.last_gaze = None
        self.last_saccade_time = time.time()
        self.BASELINE_PUPIL_SIZE = 20.0
        
        # For prediction smoothing
        self.prediction_history = deque(maxlen=30)
        self.engagement_labels = ['Disengaged', 'Partially Engaged', 'Fully Engaged']
        
        # Tracking state
        self.running = False
        self.cap = None

    def _preprocess_features(self, features):
        """Convert features to tensor format for model input"""
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
        
        return torch.FloatTensor(feature_vector).to(self.device)

    def predict_engagement(self, features):
        """Predict engagement level using the loaded model or fallback logic"""
        if self.model_loaded and self.model is not None:
            with torch.no_grad():
                input_tensor = self._preprocess_features(features)
                logits = self.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()

                self.prediction_history.append(prediction)

                if len(self.prediction_history) >= 30:  # Use window size of 30
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
        else:
            # Fallback logic based on extracted features
            engagement_score = 0.0
            
            # Simple rules using the extracted features
            if features.time_since_gaze_shift > 1.0:
                engagement_score += 0.4
            
            if features.time_since_head_movement > 1.5:
                engagement_score += 0.3
            
            if features.eye_contact_duration > 1.0:
                engagement_score += 0.3
                
            # Determine engagement level
            if engagement_score > 0.7:
                prediction = 2  # Fully engaged
            elif engagement_score > 0.3:
                prediction = 1  # Partially engaged
            else:
                prediction = 0  # Disengaged
                
            # Simple smoothing with prediction history
            self.prediction_history.append(prediction)
            if len(self.prediction_history) >= 30:
                counts = np.bincount(list(self.prediction_history))
                smoothed_prediction = np.argmax(counts)
            else:
                smoothed_prediction = prediction
                
            return {
                'raw_prediction': self.engagement_labels[prediction],
                'smoothed_prediction': self.engagement_labels[smoothed_prediction],
                'confidence': 0.8,  # Placeholder confidence
                'probabilities': [0.1, 0.1, 0.8]  # Placeholder
            }

    def start(self, camera_id=0):
        """Start the webcam capture"""
        if self.running:
            return True
            
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False
            
        self.running = True
        return True
        
    def stop(self):
        """Stop the webcam capture"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def process_frame(self):
        """Process a single frame from the webcam"""
        if not self.running or not self.cap:
            return None, {}
            
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return None, {}
            
        # Flip the frame horizontally for a more natural view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(frame_rgb)
        
        # Create empty overlay
        overlay = np.zeros_like(frame)
        frame_h, frame_w = frame.shape[:2]
        
        # Default values
        state = "Unknown"
        fixation_avg = 0
        pupil_mean = 0
        blink_rate = 0
        engagement_data = {}
        
        # If no face detected, consider distracted
        if not results.multi_face_landmarks:
            self.concentration_updated.emit(0.0)  # Signal concentration as 0.0 (no face detected)
            # Create anonymized view even without face
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(frame_gray, 50, 255, cv2.THRESH_BINARY_INV)
            anonymized = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            final_output = anonymized
            
            # Add text showing no face detected
            cv2.rectangle(final_output, (0, 0), (420, 230), (0, 0, 0), -1)
            cv2.putText(final_output, "No Face Detected", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
            
            return final_output, {}
        
        # Face detected - continue with processing
        for face_landmarks in results.multi_face_landmarks:
            # Extract eye and iris indices
            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]
            left_iris_idx = [468, 469, 470, 471, 472]
            right_iris_idx = [473, 474, 475, 476, 477]

            # Calculate eye aspect ratio
            def eye_aspect_ratio(landmarks, indices):
                p = [landmarks[i] for i in indices]
                coords = [(int(p_.x * frame_w), int(p_.y * frame_h)) for p_ in p]
                vert1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
                vert2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
                hor = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
                return (vert1 + vert2) / (2.0 * hor)

            # Calculate pupil diameter
            def pupil_diameter(landmarks, indices):
                left = landmarks[indices[0]]
                right = landmarks[indices[3]]
                return np.sqrt((left.x - right.x)**2 + (left.y - right.y)**2) * frame_w
            
            # Calculate eye metrics
            ear_left = eye_aspect_ratio(face_landmarks.landmark, left_eye_idx)
            ear_right = eye_aspect_ratio(face_landmarks.landmark, right_eye_idx)
            ear_avg = (ear_left + ear_right) / 2
            
            # Detect blinks
            if ear_avg < 0.20:
                self.blink_timestamps.append(time.time())
            
            now = time.time()
            recent_blinks = [t for t in self.blink_timestamps if now - t < 10]
            blink_rate = len(recent_blinks) / 10
            
            # Calculate pupil sizes
            pupil_left = pupil_diameter(face_landmarks.landmark, left_iris_idx)
            pupil_right = pupil_diameter(face_landmarks.landmark, right_iris_idx)
            pupil_avg = (pupil_left + pupil_right) / 2
            self.pupil_sizes.append(pupil_avg)
            
            # Track gaze movements
            current_gaze = (
                (face_landmarks.landmark[468].x + face_landmarks.landmark[473].x) * frame_w / 2,
                (face_landmarks.landmark[468].y + face_landmarks.landmark[473].y) * frame_h / 2
            )
            
            if self.last_gaze:
                dx = current_gaze[0] - self.last_gaze[0]
                dy = current_gaze[1] - self.last_gaze[1]
                amplitude = np.sqrt(dx ** 2 + dy ** 2)
                if amplitude > 5:
                    self.saccade_buffer.append({'amplitude': amplitude, 'timestamp': now})
                    self.fixation_durations.append(now - self.last_saccade_time)
                    self.last_saccade_time = now
            self.last_gaze = current_gaze
            
            # Calculate average fixation duration
            fixation_avg = np.mean(self.fixation_durations) * 1000 if self.fixation_durations else 0
            pupil_mean = np.mean(self.pupil_sizes) if self.pupil_sizes else 0
            
            # Compute distraction score based on physiology
            pupil_std_ratio = np.std(self.pupil_sizes) / (self.BASELINE_PUPIL_SIZE + 1e-5) if self.pupil_sizes else 0
            distraction_score = sum([
                blink_rate > 0.6,
                fixation_avg < 80,
            ])
            
            # Extract features using the existing FaceFeatureExtractor
            features = self.feature_extractor.extract_features(frame, face_landmarks)
            
            # Predict engagement using model or fallback method
            engagement_data = self.predict_engagement(features)
            
            # Determine focus state based on engagement prediction and distraction score
            if engagement_data['smoothed_prediction'] == 'Fully Engaged' and distraction_score <= 1:
                state = "Focused"
                self.concentration_updated.emit(1.0)  # Signal concentration as 1.0 (focused)
            else:
                state = "Distracted"
                self.concentration_updated.emit(0.0)  # Signal concentration as 0.0 (distracted)
            
            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                image=overlay,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
            )
        
        # Create final visualization
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(frame_gray, 50, 255, cv2.THRESH_BINARY_INV)
        anonymized = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        final_output = cv2.addWeighted(anonymized, 1.0, overlay, 1.0, 0)
        
        # Add metrics to visualization
        cv2.rectangle(final_output, (0, 0), (420, 230), (0, 0, 0), -1)
        cv2.putText(final_output, f"Blink Rate: {blink_rate:.2f}/s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(final_output, f"Fixation: {fixation_avg:.0f}ms", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(final_output, f"Pupil: {pupil_mean:.1f}px", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Also display engagement level
        engagement_text = engagement_data.get('smoothed_prediction', 'Unknown')
        cv2.putText(final_output, f"Engagement: {engagement_text}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Set color based on focus state
        state_color = (0, 255, 0) if state == "Focused" else (0, 0, 255)
        cv2.putText(final_output, state, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.8, state_color, 4)
        
        # Return the final visualization and engagement data
        return final_output, engagement_data 