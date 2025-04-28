# eyeGesturesWrapper.py

from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3

class EyeGestures:
    def __init__(self, camera_index=0, screen_width=500, screen_height=500):
        self.gestures = EyeGestures_v3()
        self.cap = VideoCapture(camera_index)
        self.calibrate = True  # Start with calibration enabled
        self.screen_width = screen_width
        self.screen_height = screen_height

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        event, cevent = self.gestures.step(
            frame,
            self.calibrate,
            self.screen_width,
            self.screen_height,
            context="student_ui"
        )

        if event:
            norm_x = event.point[0]
            norm_y = event.point[1]
            return norm_x, norm_y
        else:
            return None, None

    def stop(self):
        self.cap.release()
