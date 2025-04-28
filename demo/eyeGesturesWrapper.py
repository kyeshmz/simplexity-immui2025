from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3

class EyeGestures:
    def __init__(self):
        self.gestures = EyeGestures_v3()
        self.cap = VideoCapture(0)
        self.screen_width = 500
        self.screen_height = 500

    def process_frame(self, calibrating=True):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        event, cevent = self.gestures.step(
            frame,
            calibrating,  # 👈 pass calibration flag properly
            self.screen_width,
            self.screen_height,
            context="my_context"
        )

        if event:
            return event.point[0], event.point[1]
        else:
            return None, None
