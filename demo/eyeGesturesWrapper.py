from eyeGestures.utils import VideoCapture
from eyeGestures import EyeGestures_v3

class EyeGestures:
    def __init__(self):
        self.gestures = EyeGestures_v3()
        self.cap = VideoCapture(0)
        self.screen_width = 500
        self.screen_height = 500
        self.last_valid_x = 0.5
        self.last_valid_y = 0.5

    def process_frame(self, screen_width=500, screen_height=500, calibrating=True):
        ret, frame = self.cap.read()
        if not ret:
            return self.last_valid_x, self.last_valid_y

        event, cevent = self.gestures.step(
            frame,
            calibrating,
            self.screen_width,
            self.screen_height,
            context="my_context"
        )

        if event:
            # Normalize to [0,1] if needed
            x, y = event.point[0] / self.screen_width, event.point[1] / self.screen_height
            self.last_valid_x = x
            self.last_valid_y = y
            return x, y
        else:
            # No new event: reuse last valid gaze
            return self.last_valid_x, self.last_valid_y