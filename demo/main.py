#!/usr/bin/env python3
# Main application code

import sys
import threading
import argparse
import time
import os

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor
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
    # Optional: Start Student UI listener thread if needed for /setConcentration
    # student_osc_thread = threading.Thread(
    #    target=run_osc_server,
    #    args=(OSC_IP, OSC_PORT, student_osc_handler, "Student UI Listener"),
    #    daemon=True)
    # student_osc_thread.start()

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
        # sys.exit(1)  # Optionally exit

    dashboard_window = DashboardWindow(dashboard_osc_handler)

    # --- Webcam Window Setup ---
    # Check if model exists
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Using fallback engagement detection.")
    else:
        print(f"Using engagement model: {model_path}")
    
    # Initialize the webcam processor with model path
    webcam_processor = WebcamProcessor(model_path=model_path)
    
    # Connect webcam concentration updates to OSC handler
    webcam_processor.concentration_updated.connect(student_osc_handler.concentration_received)
    
    # Initialize and show webcam viewer if requested
    webcam_window = None
    if args.show_webcam:
        print("Webcam window requested via command line flag.")
        webcam_window = WebcamViewerWidget(webcam_processor)
        webcam_window.show()
        # Start webcam processing
        webcam_window.start()
    else:
        # If not showing the window, still start the webcam processor in the background
        # This way it will still send concentration updates without showing UI
        webcam_processor.start()

    # --- Show Windows ---
    student_window.show()
    dashboard_window.show()  # Show the dashboard window

    # Run the application
    exit_code = app.exec()
    
    # Ensure webcam is properly closed on exit
    if webcam_processor is not None:
        webcam_processor.stop()
    
    sys.exit(exit_code) 