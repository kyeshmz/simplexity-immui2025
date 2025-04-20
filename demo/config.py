import os

# --- Configuration ---
# Student UI / Annotation Config
STUDENT_ID = f"Student_{os.getpid()}"  # Unique ID for this instance
OSC_IP = "127.0.0.1"  # Listen on localhost
OSC_PORT = 5005       # Port for student UI to listen on
OSC_TOGGLE_ADDRESS = "/toggleAnnotations"  # Global ON/OFF for this student UI
OSC_DISPLAY_ADDRESS = "/displayAnnotation"  # Trigger specific word popup for this student UI
# OSC Address to receive simulated concentration for this student UI (optional)
OSC_SET_CONCENTRATION = "/setConcentration"

# Dashboard Config (Where Student UI Sends Data To)
DASHBOARD_IP = "127.0.0.1"
DASHBOARD_PORT = 5006  # Port Dashboard listens on

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
}

PARAGRAPH_HOVER_CONTENT = {
    1: {
        1: {
            "text": "A microcontroller pin set as “input” should be connected to a pull-down resistor (e.g. 10kΩ to ground) to stabilize voltage levels, ensuring clean signal detection. Misconfiguring this could create accidental short circuits.",
            "image": "images/para1_focused.jpg"
        },
        2: {
            "text": "Imagine a water pipe system. If a valve (input pin) is opened to let water flow into a drain (ground) but the pump (current source) isn’t turned on, water stagnates - like how a floating input pin picks up erratic signals.",
            "image": "images/para1_distracted.jpg"
        }
    },
    2: {
        1: {
            "text": "[Placeholder text explanation for Focused User]",
            "image": "images/para2_focused.jpg"
        },
        2: {
            "text": "[Placeholder text explanation Distracted User]",
            "image": "images/para2_distracted.jpg"
        }
    }
}




SAMPLE_PARAGRAPH = """When configuring I/O pins to receive current to ground, improper settings can create unexpected behavior.

Voltage drops can occur if components are misaligned, leading to potential circuit failure or incorrect readings."""

IMAGE_NOT_FOUND_PATH = "images/not_found.png"  # Optional: Image to show if specific word image missing
DEFAULT_POPUP_IMAGE_TEXT = "Hover or trigger word..."  # Placeholder text if needed

# Highlighting for Active Word
ACTIVE_HIGHLIGHT_COLOR = "yellow" 