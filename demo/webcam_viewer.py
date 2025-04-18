import cv2
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt, Slot

class WebcamViewerWidget(QWidget):
    def __init__(self, webcam_processor, parent=None):
        super().__init__(parent)
        self.webcam_processor = webcam_processor
        self.setWindowTitle("Simplexity Focus Monitor")
        self.setGeometry(1350, 100, 640, 480)  # Position on right of screens
        
        # Create UI components
        self.layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)
        
        # Create a timer for updating the video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Connect signals
        self.webcam_processor.frame_processed.connect(self.display_processed_frame)
        
        # Initialize frame display
        self.current_frame = None
        
    def start(self):
        """Start the webcam feed"""
        if self.webcam_processor.start():
            self.timer.start(33)  # ~30 fps
            
    def stop(self):
        """Stop the webcam feed"""
        self.timer.stop()
        self.webcam_processor.stop()
        
    def closeEvent(self, event):
        """Handle widget close event"""
        self.stop()
        super().closeEvent(event)
        
    @Slot()
    def update_frame(self):
        """Update the frame from the webcam processor"""
        frame, _ = self.webcam_processor.process_frame()
        if frame is not None:
            self.current_frame = frame
            self.update_display()
            
    @Slot(np.ndarray, dict)
    def display_processed_frame(self, frame, data):
        """Display a processed frame from external source"""
        if frame is not None:
            self.current_frame = frame
            self.update_display()
            
    def update_display(self):
        """Update the image label with the current frame"""
        if self.current_frame is None:
            return
            
        # Convert OpenCV BGR image to RGB for Qt
        rgb_image = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        
        # Convert to QImage and then to QPixmap
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale pixmap if necessary while maintaining aspect ratio
        pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
                              
        # Display the image
        self.image_label.setPixmap(pixmap) 