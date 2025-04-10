#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ===== Helper Functions =====
def eye_aspect_ratio(landmarks, indices, frame_width, frame_height):
    p = [landmarks[i] for i in indices]
    coords = [(int(p_.x * frame_width), int(p_.y * frame_height)) for p_ in p]
    vert1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    vert2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    hor = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    return (vert1 + vert2) / (2.0 * hor)

def pupil_diameter(landmarks, indices, frame_width, frame_height):
    left = landmarks[indices[0]]
    right = landmarks[indices[3]]
    return np.sqrt((left.x - right.x)**2 + (left.y - right.y)**2) * frame_width

# ========== CONFIGURATION ==========
BASELINE_PUPIL_SIZE = 20.0
ATTENTION_THRESHOLD = 5.0
BLINK_RATE_WINDOW = 10
EYE_BOX_Y = (0.35, 0.65)
EYE_BOX_X = (0.3, 0.7)

# ========== INITIALIZATION ==========
attention_history = deque(maxlen=30)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

blink_timestamps = deque(maxlen=100)
saccade_buffer = deque(maxlen=30)
fixation_durations = deque(maxlen=30)
last_gaze = None
last_saccade_time = time.time()
pupil_sizes = deque(maxlen=30)

fixation_avg = 0
pupil_mean = 0
calibration_frames = 150
calibration_pupil_sizes = []
calibrating = True
calibration_message_displayed = False

# ========== MAIN LOOP ==========
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    overlay = np.zeros_like(frame)
    frame_h, frame_w = frame.shape[:2]
    out_of_bounds_logged = False
    attention = "Calibrating"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]
            left_iris_idx = [468, 469, 470, 471, 472]
            right_iris_idx = [473, 474, 475, 476, 477]

            # Blink detection
            ear_left = eye_aspect_ratio(face_landmarks.landmark, left_eye_idx, frame_w, frame_h)
            ear_right = eye_aspect_ratio(face_landmarks.landmark, right_eye_idx, frame_w, frame_h)
            ear_avg = (ear_left + ear_right) / 2
            if ear_avg < 0.20:
                blink_timestamps.append(time.time())

            # Blink rate
            now = time.time()
            recent_blinks = [t for t in blink_timestamps if now - t < BLINK_RATE_WINDOW]
            blink_rate = len(recent_blinks) / BLINK_RATE_WINDOW

            # Pupil size
            pupil_left = pupil_diameter(face_landmarks.landmark, left_iris_idx, frame_w, frame_h)
            pupil_right = pupil_diameter(face_landmarks.landmark, right_iris_idx, frame_w, frame_h)
            pupil_avg = (pupil_left + pupil_right) / 2
            pupil_sizes.append(pupil_avg)

            # Gaze position
            current_gaze = (
                (face_landmarks.landmark[468].x + face_landmarks.landmark[473].x) * frame_w / 2,
                (face_landmarks.landmark[468].y + face_landmarks.landmark[473].y) * frame_h / 2
            )

            if last_gaze:
                dx = current_gaze[0] - last_gaze[0]
                dy = current_gaze[1] - last_gaze[1]
                amplitude = np.sqrt(dx ** 2 + dy ** 2)
                if amplitude > 5:
                    direction = "backward" if dx < 0 else "forward"
                    saccade_buffer.append({'amplitude': amplitude, 'direction': direction, 'timestamp': now})
                    fixation_durations.append(now - last_saccade_time)
                    last_saccade_time = now
            last_gaze = current_gaze

            # Attention estimation
            if calibrating:
                calibration_pupil_sizes.append(pupil_avg)
                if not calibration_message_displayed:
                    print("[CALIBRATION] Gathering baseline pupil data...")
                    calibration_message_displayed = True
                if len(calibration_pupil_sizes) >= calibration_frames:
                    BASELINE_PUPIL_SIZE = np.mean(calibration_pupil_sizes)
                    print(f"[CALIBRATION COMPLETE] Baseline pupil size set to: {BASELINE_PUPIL_SIZE:.2f}px")
                    calibrating = False
            else:
                regressions = len([s for s in saccade_buffer if s['direction'] == 'backward'])
                fixation_avg = np.mean(fixation_durations) * 1000 if fixation_durations else 0
                pupil_mean = np.mean(pupil_sizes) if pupil_sizes else 0
                pupil_std_ratio = np.std(pupil_sizes) / (BASELINE_PUPIL_SIZE + 1e-5)
                distraction_score = 0
                if blink_rate > 0.6:
                    distraction_score += 1
                if regressions > 10:
                    distraction_score += 1
                if fixation_avg < 80:
                    distraction_score += 1
                if pupil_std_ratio > 0.4:
                    distraction_score += 1
                current_state = "Focused" if distraction_score <= 1 else "Distracted"
                attention_history.append(current_state)
                attention = "Focused" if attention_history.count("Distracted") < 15 else "Distracted"

            # Eye zone check
            eye_center_x = (face_landmarks.landmark[468].x + face_landmarks.landmark[473].x) / 2
            eye_center_y = (face_landmarks.landmark[468].y + face_landmarks.landmark[473].y) / 2
            if not (EYE_BOX_X[0] <= eye_center_x <= EYE_BOX_X[1] and EYE_BOX_Y[0] <= eye_center_y <= EYE_BOX_Y[1]):
                if not out_of_bounds_logged:
                    print("[LOG] Eyes out of designated zone")
                    out_of_bounds_logged = True

            # Draw face mesh
            mp_drawing.draw_landmarks(
                image=overlay,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
            )

    # ========== THRESHOLD ANONYMIZATION ==========
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(frame_gray, 50, 255, cv2.THRESH_BINARY_INV)
    anonymized = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    final_output = cv2.addWeighted(anonymized, 1.0, overlay, 1.0, 0)

    # ========== DRAW METRICS ==========
    if not calibrating:
        cv2.rectangle(final_output, (0, 0), (360, 230), (0, 0, 0), -1)
        cv2.putText(final_output, f"Blink Rate: {blink_rate:.2f}/s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(final_output, f"Fixation: {fixation_avg:.0f}ms", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(final_output, f"Pupil: {pupil_mean:.1f}px", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        state_color = (0, 255, 0) if attention == "Focused" else (0, 0, 255)
        cv2.putText(final_output, attention, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.8, state_color, 4)
    else:
        progress = len(calibration_pupil_sizes) / calibration_frames
        cv2.rectangle(frame, (0, 0), (360, 100), (0, 0, 0), -1)
        cv2.putText(frame, "Calibrating...", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.rectangle(frame, (10, 60), (350, 80), (255, 255, 255), 2)
        cv2.rectangle(frame, (12, 62), (int(12 + (336 * progress)), 78), (255, 255, 255), -1)
        final_output[0:100, 0:360] = frame[0:100, 0:360]

    cv2.imshow("Simplexity v2", final_output)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
