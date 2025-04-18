import os
from PySide6.QtGui import QColor

# --- Configuration ---
# Student UI / Annotation Config
# Use environment variable if set, otherwise generate unique ID
STUDENT_ID = os.environ.get("STUDENT_ID", f"Student_{os.getpid()}") 
OSC_IP = "127.0.0.1"  # Listen on localhost
OSC_PORT = 5005       # Port for student UI to listen on
OSC_TOGGLE_ADDRESS = "/toggleAnnotations" # Global ON/OFF for this student UI
OSC_DISPLAY_ADDRESS = "/displayAnnotation"  # Trigger specific word popup for this student UI
# OSC Address to receive simulated concentration for this student UI (optional)
OSC_SET_CONCENTRATION = "/setConcentration" 

# Dashboard Config (Where Student UI Sends Data To)
DASHBOARD_IP = "127.0.0.1"
DASHBOARD_PORT = 5006 # Port Dashboard listens on

# OSC Addresses for Sending Data *TO* Dashboard
DASHBOARD_STATUS_ADDRESS = "/student/status"
DASHBOARD_TRIGGER_ADDRESS = "/student/trigger"

# Updated structure: word -> {text: explanation, image: path}
ANNOTATION_WORDS = {
    "configuring":   {"text": "Setting up or arranging parameters for operation.", "image": "images/configuring.png"},
    "I/O":           {"text": "Input/Output, refers to communication between a system and the outside world.", "image": "images/io.png"},
    "pins":          {"text": "Physical connection points on an integrated circuit or connector.", "image": "images/pins.png"},
    "current":       {"text": "The flow of electric charge.", "image": "images/current.png"},
    "ground":        {"text": "A common reference point in an electrical circuit from which voltages are measured.", "image": "images/ground.png"}
    # Add more words like 'improper', 'settings', 'unexpected', 'behavior' if needed
}
SAMPLE_PARAGRAPH = "When configuring I/O pins to recieve current to ground, improper settings can create unexpected behavior. "


IMAGE_NOT_FOUND_PATH = "images/not_found.png" # Optional: Image to show if specific word image missing
DEFAULT_POPUP_IMAGE_TEXT = "Hover or trigger word..." # Placeholder text if needed

# Highlighting for Active Word
ACTIVE_HIGHLIGHT_COLOR = QColor("yellow")

# Dashboard Color mapping for BINARY concentration levels
DASHBOARD_CONCENTRATION_COLORS = {
    0: QColor("orange"), # Not concentrated
    1: QColor("lightgreen") # Concentrated
}

# Charting settings
MAX_CHART_DATA_POINTS = 100 