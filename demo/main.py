#!/usr/bin/env python3
# Main application code

import sys
import threading
import argparse
import time
import os

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor
from PySide6.QtCore import QTimer
import pyqtgraph as pg

# Import configuration
from config import (
    STUDENT_ID, OSC_IP, OSC_PORT,
    DASHBOARD_IP, DASHBOARD_PORT, SAMPLE_PARAGRAPH
)

# Import custom components
from osc_handler import OSCHandler
from dashboard_osc_handler import DashboardOSCHandler
from main_window import MainWindow
from dashboard_window import DashboardWindow
from osc_utils import run_osc_server

# Import webcam components
from webcam_processor import WebcamProcessor
from webcam_viewer import WebcamViewerWidget

# --- Main Execution ---
if __name__ == "__main__":
    # Set background for plots (optional, 'w' is white)
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')  # 'k' is black

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Student Annotation UI with optional Webcam Viewer.')
    parser.add_argument('--show-webcam', action='store_true', help='Show the webcam processing window.')
    parser.add_argument('--model-path', default='best_model_v3.pth', help='Path to the engagement classifier model')
    args = parser.parse_args()
    # --- End Argument Parsing ---

    app = QApplication(sys.argv)

    # --- Student UI Setup ---
    student_osc_handler = OSCHandler()
    student_window = MainWindow(student_osc_handler, STUDENT_ID, DASHBOARD_IP, DASHBOARD_PORT)

    # --- Dashboard Setup ---
    dashboard_osc_handler = DashboardOSCHandler()
    dashboard_osc_thread = threading.Thread(
        target=run_osc_server,
        args=(DASHBOARD_IP, DASHBOARD_PORT, dashboard_osc_handler, "Dashboard Listener"),
        daemon=True)
    dashboard_osc_thread.start()

    if not dashboard_osc_thread.is_alive():
        print("Dashboard OSC server thread failed to start.")

    dashboard_window = DashboardWindow(dashboard_osc_handler)

    # --- Webcam Processor Setup ---
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Using fallback engagement detection.")
    else:
        print(f"Using engagement model: {model_path}")

    # Pass the student ID to the webcam processor for OSC messaging
    webcam_processor = WebcamProcessor(model_path=model_path, student_id=STUDENT_ID)
    webcam_processor.concentration_updated.connect(student_osc_handler.concentration_received)

    # --- Start Webcam Processor ---
    webcam_processor.start()

    frame_timer = QTimer()
    frame_timer.timeout.connect(webcam_processor.process_frame)
    frame_timer.start(30)

    # Create webcam viewer widget
    webcam_viewer = WebcamViewerWidget(webcam_processor)

    # --- Show Windows ---
    screen = app.primaryScreen()
    available_geometry = screen.availableGeometry()
    
    # Set window dimensions
    full_width = available_geometry.width()
    full_height = available_geometry.height()
    left_half_width = full_width // 2
    right_half_width = full_width - left_half_width
    top_left_height = full_height // 2
    bottom_left_height = full_height - top_left_height

    # Position student window on the right side
    student_window.setGeometry(
        available_geometry.left() + left_half_width, 
        available_geometry.top(), 
        right_half_width, 
        full_height
    )
    student_window.show()

    # Position webcam viewer on the top left
    webcam_viewer.setGeometry(
        available_geometry.left(), 
        available_geometry.top(), 
        left_half_width, 
        top_left_height
    )
    webcam_viewer.show()

    # Position dashboard window on the bottom left
    dashboard_window.setGeometry(
        available_geometry.left(), 
        available_geometry.top() + top_left_height, 
        left_half_width, 
        bottom_left_height
    )
    dashboard_window.show()

    # Run the application
    exit_code = app.exec()

    # Ensure webcam is properly closed on exit
    if webcam_processor is not None:
        webcam_processor.stop()

    sys.exit(exit_code)
