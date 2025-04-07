#!/usr/bin/env python3
import cv2
import numpy as np
from gaze_tracking import GazeTracking
import time


class EnhancedConfusionDetector:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.saccade_buffer = []
        self.last_gaze = None
        self.last_saccade_time = time.time()
        self.fixation_durations = []

    def update(self, current_gaze):
        if self.last_gaze is not None:
            dx = current_gaze[0] - self.last_gaze[0]
            dy = current_gaze[1] - self.last_gaze[1]
            amplitude = np.sqrt(dx ** 2 + dy ** 2)

            if amplitude > 5:  # Saccade detection threshold (pixels)
                direction = "backward" if dx < 0 else "forward"
                self.saccade_buffer.append({
                    'amplitude': amplitude,
                    'direction': direction,
                    'timestamp': time.time()
                })

                # Calculate fixation duration
                now = time.time()
                self.fixation_durations.append(now - self.last_saccade_time)
                self.last_saccade_time = now

                # Maintain window size
                if len(self.saccade_buffer) > self.window_size:
                    self.saccade_buffer.pop(0)
                    self.fixation_durations.pop(0)

        self.last_gaze = current_gaze

    def get_metrics(self):
        metrics = {
            'fixation_avg': np.mean(self.fixation_durations) * 1000 if self.fixation_durations else 0,
            'saccade_amp': self.saccade_buffer[-1]['amplitude'] if self.saccade_buffer else 0,
            'regressions': len([s for s in self.saccade_buffer if s['direction'] == 'backward'])
        }
        metrics['confused'] = (metrics['regressions'] > 3) and \
                              (np.std(
                                  [s['amplitude'] for s in self.saccade_buffer]) > 15) if self.saccade_buffer else False
        return metrics


# Initialize components
gaze = GazeTracking()
detector = EnhancedConfusionDetector()
webcam = cv2.VideoCapture(0)

# Configure text properties
STATUS_FONT = cv2.FONT_HERSHEY_SIMPLEX
STATUS_SCALE = 3
STATUS_COLOR = (0, 255, 0)  # Green for focused
STATUS_THICKNESS = 5

METRICS_FONT = cv2.FONT_HERSHEY_DUPLEX
METRICS_SCALE = 0.8
METRICS_COLOR = (200, 200, 200)  # Light gray
METRICS_THICKNESS = 1

try:
    while True:
        _, frame = webcam.read()
        gaze.refresh(frame)

        pupil_left = gaze.pupil_left_coords()
        pupil_right = gaze.pupil_right_coords()
        current_gaze = pupil_left or pupil_right

        if current_gaze:
            detector.update(current_gaze)
            metrics = detector.get_metrics()

            # Get frame dimensions
            height, width = frame.shape[:2]

            # Display confusion status (centered at top)
            status = "CONFUSED" if metrics['confused'] else "NOT CONFUSED"
            color = (0, 0, 255) if metrics['confused'] else (0, 255, 0)

            (text_width, text_height), _ = cv2.getTextSize(status, STATUS_FONT, STATUS_SCALE, STATUS_THICKNESS)
            cv2.putText(frame, status,
                        ((width - text_width) // 2, text_height + 20),
                        STATUS_FONT, STATUS_SCALE, color, STATUS_THICKNESS)

            # Display metrics (bottom left)
            metrics_text = [
                f"Fixation: {metrics['fixation_avg']:.0f}ms",
                f"Saccade: {metrics['saccade_amp']:.1f}px",
                f"Regressions: {metrics['regressions']}"
            ]

            y_start = height - 30
            for i, text in enumerate(reversed(metrics_text)):
                cv2.putText(frame, text, (20, y_start - (i * 30)),
                            METRICS_FONT, METRICS_SCALE, METRICS_COLOR, METRICS_THICKNESS)

        # Show annotated frame
        frame = gaze.annotated_frame()
        cv2.imshow("Simplexity v1: Confusion Detector", frame)

        if cv2.waitKey(1) == 27:
            break
finally:
    webcam.release()
    cv2.destroyAllWindows()
