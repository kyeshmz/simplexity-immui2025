# Placeholder for main application code 

import sys
import threading
import os # Added for path joining
import time # For chart data
from PySide6.QtWidgets import (QApplication, QMainWindow, QTextEdit, 
                               QVBoxLayout, QWidget, QPushButton, QHBoxLayout, 
                               QLabel, QFrame, QStatusBar, QScrollArea,
                               QSlider, QLineEdit, QFormLayout, QSizePolicy # Added several
)
from PySide6.QtCore import Qt, Signal, Slot, QObject, QPoint, QTimer # Added QTimer
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QTextDocument, QPixmap, QMouseEvent, QFontMetrics

from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient

# --- Charting Imports ---
import pyqtgraph as pg
from pyqtgraph import PlotWidget, mkPen

# --- Configuration ---
# Student UI / Annotation Config
STUDENT_ID = f"Student_{os.getpid()}" # Unique ID for this instance
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

# --- OSC Communication Handler (for Student UI) ---
class OSCHandler(QObject):
    annotation_state_changed = Signal(bool)
    display_annotation_requested = Signal(str)
    # Signal to update concentration based on external OSC message (optional)
    concentration_received = Signal(float) 

    def __init__(self):
        super().__init__()
        self.show_annotations = False # Global enabled state for this student UI

    def handle_message(self, address, *args):
        """Handles incoming OSC messages for this student UI."""
        print(f"Student UI Received: {address} {args}")

        if address == OSC_TOGGLE_ADDRESS:
            try:
                new_state = bool(int(args[0]))
                if new_state != self.show_annotations:
                    self.show_annotations = new_state
                    self.annotation_state_changed.emit(self.show_annotations)
                    print(f"Student UI Annotation state changed via OSC to: {self.show_annotations}")
            except (IndexError, ValueError) as e:
                print(f"Error processing {OSC_TOGGLE_ADDRESS} message ({args}): {e}.")

        elif address == OSC_DISPLAY_ADDRESS:
            try:
                word_to_display = str(args[0]) if args else ""
                print(f"Student UI OSC request to display annotation for: '{word_to_display}'")
                self.display_annotation_requested.emit(word_to_display)
            except Exception as e:
                 print(f"Error processing {OSC_DISPLAY_ADDRESS} message ({args}): {e}")

        elif address == OSC_SET_CONCENTRATION: # Handle external concentration input
            try:
                conc_float = float(args[0])
                # Clamp value between 0.0 and 1.0
                conc_float = max(0.0, min(1.0, conc_float))
                self.concentration_received.emit(conc_float)
                print(f"Student UI Received concentration: {conc_float}")
            except (IndexError, ValueError) as e:
                print(f"Error processing {OSC_SET_CONCENTRATION} message ({args}): {e}. Expected float.")


# --- Custom Annotation Popup Widget ---
class AnnotationPopup(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Window flags to make it look like a tooltip/popup
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        # Layout and content widgets
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(100, 100) # Min size for image area
        self.text_label = QLabel()
        self.text_label.setWordWrap(True)
        self.text_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.text_label)

        # Styling (optional, customize as needed)
        self.setStyleSheet("""
            AnnotationPopup {
                background-color: #ffffff;
                border: 1px solid #aaaaaa;
                border-radius: 4px;
            }
            QLabel {
                color: #333333;
                background-color: transparent; /* Ensure labels don't obscure frame background */
            }
        """)

    def setContent(self, text, image_path):
        # Set Text
        self.text_label.setText(text)
        self.text_label.setVisible(bool(text))

        # Set Image
        pixmap = None
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                 print(f"Error loading image: {image_path}")
                 pixmap = None # Failed to load
        elif image_path:
             print(f"Image file not found: {image_path}")

        if pixmap:
            # Simple scaling logic (adjust as needed)
            max_width = 200
            if pixmap.width() > max_width:
                pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.image_label.setVisible(True)
            self.image_label.setFixedSize(pixmap.size()) # Adjust label size to pixmap
        else:
            self.image_label.clear()
            self.image_label.setVisible(False)
            self.image_label.setFixedSize(0,0) # Collapse if no image

        # Adjust the popup size to fit content
        self.adjustSize()

    def showAt(self, pos: QPoint):
        # Ensure position is reasonable (within screen bounds if possible)
        # Basic check - more sophisticated checks might be needed
        screen_geo = QApplication.primaryScreen().availableGeometry()
        popup_rect = self.geometry()
        popup_rect.moveTopLeft(pos)

        # Adjust if going off-screen (simple adjustment)
        if popup_rect.right() > screen_geo.right():
            pos.setX(screen_geo.right() - popup_rect.width() - 5)
        if popup_rect.bottom() > screen_geo.bottom():
            pos.setY(screen_geo.bottom() - popup_rect.height() - 5)
        if popup_rect.left() < screen_geo.left():
            pos.setX(screen_geo.left() + 5)
        if popup_rect.top() < screen_geo.top():
            pos.setY(screen_geo.top() + 5)


        self.move(pos)
        self.show()

# --- Custom Text Edit for Hover Detection ---
class HoverTextEdit(QTextEdit):
    # Emits the word being hovered over (or empty string) and the global mouse position
    word_hovered = Signal(str, QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.last_hovered_word = ""

    def mouseMoveEvent(self, event: QMouseEvent):
        cursor = self.cursorForPosition(event.pos())
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        selected_word = cursor.selectedText().strip('.,?!;:"\'()')

        current_word = ""
        if selected_word and selected_word in ANNOTATION_WORDS:
            current_word = selected_word

        # Emit word and global position if word changes
        if current_word != self.last_hovered_word:
            self.word_hovered.emit(current_word, event.globalPos())
            self.last_hovered_word = current_word
        # Also emit if mouse moves significantly over the *same* word, so popup follows
        elif current_word:
            # You might want to add a distance threshold check here
            # to avoid emitting too frequently
            self.word_hovered.emit(current_word, event.globalPos())


        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self.last_hovered_word != "":
            self.last_hovered_word = ""
            self.word_hovered.emit("", QPoint()) # Emit empty word, position irrelevant
        super().leaveEvent(event)

# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self, osc_handler, student_id):
        super().__init__()
        self.osc_handler = osc_handler
        self.student_id = student_id
        self.setWindowTitle(f"Student Annotation UI - {self.student_id}")
        self.setGeometry(100, 100, 700, 600) # Increased height for concentration controls

        # OSC Client to send data to Dashboard
        try:
            self.osc_client = SimpleUDPClient(DASHBOARD_IP, DASHBOARD_PORT)
            print(f"Student UI OSC Client configured to send to {DASHBOARD_IP}:{DASHBOARD_PORT}")
        except Exception as e:
            print(f"Error creating OSC Client: {e}. Dashboard messages will not be sent.")
            self.osc_client = None
        print(f"[Confirmation] self.osc_client is {'NOT None' if self.osc_client else 'None'}")

        # Main layout container
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create the single annotation popup (reused)
        self.annotation_popup = AnnotationPopup()

        # Text Area (using custom HoverTextEdit)
        self.text_edit = HoverTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setText(SAMPLE_PARAGRAPH)
        font = self.text_edit.font()
        font.setPointSize(font.pointSize() + 2)
        self.text_edit.setFont(font)
        self.main_layout.addWidget(self.text_edit)

        # --- Control Layout ---
        self.control_layout = QHBoxLayout()

        # Annotation Toggle Button
        self.toggle_button = QPushButton("Toggle Annotations (OFF)")
        self.control_layout.addWidget(self.toggle_button)
        self.control_layout.addStretch(1) # Push controls apart

        # Concentration Simulation Controls
        self.concentration_layout = QFormLayout()
        self.concentration_slider = QSlider(Qt.Orientation.Horizontal)
        self.concentration_slider.setRange(0, 100)
        self.concentration_slider.setValue(50) # Default value
        self.concentration_label = QLabel(f"{self.concentration_slider.value()/100.0:.2f}")
        self.concentration_layout.addRow(QLabel("Set Concentration:"), self.concentration_slider)
        self.concentration_layout.addRow(QLabel("Current Value:"), self.concentration_label)

        self.control_layout.addLayout(self.concentration_layout)

        # Add a test trigger button
        self.test_trigger_button = QPushButton("Send Test Trigger")
        self.control_layout.addWidget(self.test_trigger_button)

        self.control_layout.addStretch(1) # Push controls apart

        self.main_layout.addLayout(self.control_layout)

        # Add a status bar for global state info
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Annotations OFF")

        # --- Highlighting State ---
        self.active_highlight_cursor = None
        self.active_highlight_original_format = None
        self.highlight_format = QTextCharFormat()
        self.highlight_format.setBackground(ACTIVE_HIGHLIGHT_COLOR)

        # --- Connect Signals ---
        self.osc_handler.annotation_state_changed.connect(self.update_annotation_state)
        self.toggle_button.clicked.connect(self.toggle_annotations)
        # Hover signal triggers popup display via mouse position
        self.text_edit.word_hovered.connect(self.handle_hover_event)
        # OSC display request triggers popup display via word search
        self.osc_handler.display_annotation_requested.connect(self.show_popup_via_osc)
        
        # Concentration Slider connections
        self.concentration_slider.valueChanged.connect(self.update_concentration_display)
        self.concentration_slider.sliderReleased.connect(self.send_concentration_update) # Send when user releases slider
        # Connect external OSC concentration signal
        self.osc_handler.concentration_received.connect(self.set_concentration_from_osc)

        # Connect test trigger button
        self.test_trigger_button.clicked.connect(self.send_test_trigger)

        # Apply initial state
        self.update_annotation_state(self.osc_handler.show_annotations)
        self.send_concentration_update() # Send initial concentration

    @Slot()
    def toggle_annotations(self):
        """Toggles the global annotation state via button press."""
        new_state = not self.osc_handler.show_annotations
        self.osc_handler.show_annotations = new_state
        print(f"Button toggled global annotations to: {new_state}")
        self.update_annotation_state(new_state)
        if not new_state:
             self.annotation_popup.hide()

    @Slot(str, QPoint) # Receives word and global mouse position
    def handle_hover_event(self, word, global_pos):
        """Shows or hides the annotation popup based on hovered word and mouse position."""
        if not self.osc_handler.show_annotations or not word:
            self.annotation_popup.hide()
            return

        annotation_data = ANNOTATION_WORDS.get(word)
        if not annotation_data:
            self.annotation_popup.hide()
            return

        text = annotation_data.get('text', '')
        image_path = annotation_data.get('image')

        # Update popup content and show it near the cursor
        self.annotation_popup.setContent(text, image_path)
        # Offset position slightly below and to the right of the cursor
        self.annotation_popup.showAt(global_pos + QPoint(15, 15))

        # Find the word under the cursor again to highlight it
        local_pos = self.text_edit.mapFromGlobal(global_pos)
        cursor_at_hover = self.text_edit.cursorForPosition(local_pos)
        cursor_at_hover.select(QTextCursor.SelectionType.WordUnderCursor)
        selected_text = cursor_at_hover.selectedText().strip('.,?!;:"\'()')
        print(f"[Debug Hover Check] Expected: '{word}', Found under cursor: '{selected_text}', Match: {selected_text == word}")
        # Verify it's the correct word before highlighting
        if selected_text == word:
            self.apply_highlight(cursor_at_hover)

    @Slot(str) # Receives word to display from OSC
    def show_popup_via_osc(self, word):
        """Shows annotation popup for a specific word (found in text) if global state is ON."""
        if not self.osc_handler.show_annotations or not word:
            self.annotation_popup.hide()
            return

        annotation_data = ANNOTATION_WORDS.get(word)
        if not annotation_data:
            print(f"No annotation defined for word: '{word}'")
            self.annotation_popup.hide()
            return

        # Find the first occurrence of the word in the text edit
        doc = self.text_edit.document()
        cursor = doc.find(word, 0, QTextDocument.FindFlag.FindWholeWords)

        if cursor.isNull():
            print(f"Word '{word}' not found in the current text.")
            self.annotation_popup.hide()
            return

        # Calculate position for the popup based on the word's rectangle
        word_rect = self.text_edit.cursorRect(cursor)
        # Map the top-left of the word's rect to global coordinates
        global_pos_at_word = self.text_edit.mapToGlobal(word_rect.topLeft())

        # Position popup below the word
        popup_pos = global_pos_at_word + QPoint(0, word_rect.height() + 5)

        text = annotation_data.get('text', '')
        image_path = annotation_data.get('image')

        if isinstance(position_or_cursor, QPoint): # From hover event
            popup_pos = position_or_cursor + QPoint(15, 15)
            # Find the word under the cursor again to highlight it
            local_pos = self.text_edit.mapFromGlobal(position_or_cursor)
            cursor_at_hover = self.text_edit.cursorForPosition(local_pos)
            cursor_at_hover.select(QTextCursor.SelectionType.WordUnderCursor)
            selected_text = cursor_at_hover.selectedText().strip('.,?!;:"\'()')
            # ---> ADDED DEBUG PRINT <--- #
            print(f"[Debug Hover Check] Expected: '{word}', Found under cursor: '{selected_text}', Match: {selected_text == word}")
            # Verify it's the correct word before highlighting
            if selected_text == word:
                 target_cursor = cursor_at_hover

        elif isinstance(position_or_cursor, QTextCursor): # From OSC trigger
            target_cursor = position_or_cursor

        if target_cursor:
            self.apply_highlight(target_cursor) # Apply highlight
            self.annotation_popup.setContent(text, image_path)
            self.annotation_popup.showAt(popup_pos)

            # Send trigger message to dashboard
            print(f"[Debug] Attempting to send trigger for '{word}'. OSC Client exists: {self.osc_client is not None}") # DEBUG
            if self.osc_client:
                try:
                    print(f"[Debug] Calling send_message({DASHBOARD_TRIGGER_ADDRESS}, [{self.student_id}, {word}])") # DEBUG
                    self.osc_client.send_message(DASHBOARD_TRIGGER_ADDRESS, [self.student_id, word])
                    print(f"Sent trigger: {word} for {self.student_id}")
                except Exception as e:
                    print(f"Error sending trigger OSC: {e}")
            else:
                 print("[Debug] OSC Client is None, cannot send trigger.") # DEBUG
        else:
             print(f"[Debug] No target cursor found for word '{word}', cannot send trigger.") # DEBUG
             # Hide popup and clear highlight if we couldn't identify the word/cursor
             self.annotation_popup.hide()
             self.clear_active_highlight()

    @Slot(bool)
    def update_annotation_state(self, show):
        """Updates UI elements based on the global annotation state (button, status bar)."""
        self.toggle_button.setText(f"Toggle Annotations ({'ON' if show else 'OFF'})")
        self.status_bar.showMessage(f"Annotations {'ON' if show else 'OFF'}")
        if not show:
            self.annotation_popup.hide()
        # Clear residual formatting check (remains the same, does nothing visually now)
        if not show and self.active_highlight_original_format:
            # ... (code to clear self.active_highlight_original_format) ...
            self.active_highlight_original_format = None # Simplified

    # --- Concentration Methods ---
    @Slot(int)
    def update_concentration_display(self, value):
        """Updates the label next to the slider."""
        float_value = value / 100.0
        self.concentration_label.setText(f"{float_value:.2f}")
        # Optionally send update immediately on value change:
        # self.send_concentration_update()

    @Slot()
    def send_concentration_update(self):
        """Sends the current concentration value via OSC."""
        float_value = self.concentration_slider.value() / 100.0
        print(f"Sending concentration update for {self.student_id}: {float_value}")
        if self.osc_client:
            try:
                self.osc_client.send_message(DASHBOARD_STATUS_ADDRESS, [self.student_id, float_value, ""]) # Empty status text
            except Exception as e:
                print(f"Error sending concentration OSC: {e}")

    @Slot(float)
    def set_concentration_from_osc(self, value):
        """Updates the slider and display from an external OSC message."""
        slider_value = int(value * 100)
        self.concentration_slider.setValue(slider_value)
        self.concentration_label.setText(f"{value:.2f}")
        # We might also want to trigger sending this back to the dashboard
        # if the source wasn't the dashboard itself, depending on the desired flow.
        # For now, just update UI.

    @Slot()
    def send_test_trigger(self):
        """Manually sends a test trigger message to the dashboard."""
        test_word = "TEST_WORD"
        print(f"[Test Button] Attempting to send trigger for '{test_word}'.")
        if self.osc_client:
            try:
                self.osc_client.send_message(DASHBOARD_TRIGGER_ADDRESS, [self.student_id, test_word])
                print(f"[Test Button] Successfully sent trigger: {test_word} for {self.student_id}")
                # Also update the dashboard visually if running in same process (like here)
                # Find the dashboard window instance if possible - this is a bit hacky
                for widget in QApplication.topLevelWidgets():
                    if isinstance(widget, DashboardWindow):
                         # Manually call the slot or emit signal if preferred
                         widget.update_student_trigger(self.student_id, test_word)
                         break
            except Exception as e:
                print(f"[Test Button] Error sending trigger OSC: {e}")
        else:
             print("[Test Button] OSC Client is None, cannot send trigger.")

    # --- Highlighting Methods ---
    def clear_active_highlight(self):
        if self.active_highlight_cursor and self.active_highlight_original_format:
            # Check if cursor is still valid (might not be if text changed drastically)
            # Basic check:
            if self.active_highlight_cursor.position() < self.text_edit.document().characterCount():
                 try:
                    self.active_highlight_cursor.beginEditBlock()
                    self.active_highlight_cursor.setCharFormat(self.active_highlight_original_format)
                    self.active_highlight_cursor.endEditBlock()
                 except Exception as e:
                     print(f"Error clearing highlight: {e}") # Should be rare
            #else: print("Highlight cursor position invalid, skipping clear.") # Debug
        self.active_highlight_cursor = None
        self.active_highlight_original_format = None

    # ---> ADDED METHOD DEFINITION BACK <--- #
    def apply_highlight(self, cursor: QTextCursor):
        """Applies the highlight format to the given cursor, storing the original."""
        self.clear_active_highlight() # Clear previous before applying new
        if not cursor or cursor.isNull():
            return
        
        # Store position and format *before* applying highlight
        self.active_highlight_cursor = QTextCursor(cursor) # Make a copy
        self.active_highlight_original_format = cursor.charFormat()

        cursor.beginEditBlock()
        cursor.mergeCharFormat(self.highlight_format)
        cursor.endEditBlock()

# --- Dashboard OSC Handler (Receives data from Student UIs) ---
class DashboardOSCHandler(QObject):
    student_status_updated = Signal(str, float, str)
    student_word_triggered = Signal(str, str)

    def __init__(self): super().__init__()
    def handle_status(self, address, student_id, concentration, *args):
        status_text = args[0] if args else ""
        # print(f"Dashboard Rx Status: ID:{student_id} Conc:{concentration} Status:'{status_text}'") # Less verbose
        try: self.student_status_updated.emit(str(student_id), float(concentration), status_text)
        except ValueError: print(f"Error: Bad concentration value '{concentration}' for student {student_id}.")
        except Exception as e: print(f"Error processing status message: {e}")
    def handle_trigger(self, address, student_id, word, *args):
        # print(f"Dashboard Rx Trigger: ID:{student_id} Word:'{word}'") # Less verbose
        try: self.student_word_triggered.emit(str(student_id), str(word))
        except Exception as e: print(f"Error processing trigger message: {e}")

# --- Dashboard Student Chart Widget ---
class StudentChartWidget(QWidget):
    def __init__(self, student_id, parent=None):
        super().__init__(parent)
        self.student_id = student_id
        self.max_data_points = 100 # Max points to show on chart

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)

        # Labels
        self.id_label = QLabel(f"<b>{self.student_id}</b>")
        self.id_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trigger_label = QLabel("Last Trigger: -")
        self.trigger_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Chart
        self.plot_widget = PlotWidget()
        self.plot_widget.setMinimumHeight(100)
        self.plot_widget.setLabel('left', 'Concentration (Binary)')
        self.plot_widget.setLabel('bottom', 'Time (Updates)')
        self.plot_widget.setYRange(-0.1, 1.1) # Adjusted Y range for binary
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        # Set explicit ticks for binary values
        self.plot_widget.getAxis('left').setTicks([[(0, '0'), (1, '1')]])
        self.plot_widget.getAxis('bottom').setTicks([]) # Hide bottom ticks for simplicity
        
        # Data storage for the plot
        self.time_data = list(range(self.max_data_points)) # X-axis (indices)
        self.concentration_data = [0.0] * self.max_data_points # Y-axis

        # Create plot curve item
        self.plot_curve = self.plot_widget.plot(self.time_data, self.concentration_data, pen=mkPen('b', width=2)) # Blue line

        # Layout
        self.layout.addWidget(self.id_label)
        self.layout.addWidget(self.plot_widget)
        self.layout.addWidget(self.trigger_label)

        # Set size policy to expand horizontally but be fixed vertically
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


    def update_concentration(self, concentration):
        # ---> Convert to binary <--- #
        binary_concentration = 1.0 if concentration >= 0.5 else 0.0

        # Shift data left, add new *binary* point at the end
        self.concentration_data = self.concentration_data[1:] + [binary_concentration]
        # Update the plot curve data
        self.plot_curve.setData(self.time_data, self.concentration_data)

        # Update color based on *binary* value
        color = DashboardWindow.get_concentration_color(binary_concentration) # Pass binary value
        self.id_label.setStyleSheet(f"background-color: {color.name()}; padding: 2px; border-radius: 3px;")


    def update_trigger_word(self, word):
        self.trigger_label.setText(f"Last Trigger: {word}")

# --- Dashboard Window (Displays Charts) ---
class DashboardWindow(QMainWindow):
    # Define color mapping for BINARY concentration levels
    CONCENTRATION_COLORS = {
        0: QColor("orange"), # Not concentrated
        1: QColor("lightgreen") # Concentrated
    }
    @staticmethod
    def get_concentration_color(binary_concentration): # Input is now 0 or 1
        return DashboardWindow.CONCENTRATION_COLORS.get(int(binary_concentration), QColor("white")) # Default white

    def __init__(self, osc_handler):
        super().__init__()
        self.osc_handler = osc_handler
        self.setWindowTitle("Student Dashboard")
        self.setGeometry(900, 100, 400, 700) # Positioned to the right of main window

        # Scrollable Area Setup
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) # No horizontal scroll
        self.setCentralWidget(self.scroll_area)

        # Container widget inside scroll area
        self.container_widget = QWidget()
        self.scroll_area.setWidget(self.container_widget)

        # Layout for the container (holds student chart widgets)
        self.students_layout = QVBoxLayout(self.container_widget)
        self.students_layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Add new students at the top

        # Internal data storage: {student_id: StudentChartWidget}
        self.student_widgets = {}

        # Connect signals from OSC handler
        self.osc_handler.student_status_updated.connect(self.update_student_status)
        self.osc_handler.student_word_triggered.connect(self.update_student_trigger)

    def _get_or_create_student_widget(self, student_id):
        """Finds existing widget or creates and adds a new one with a separator."""
        if student_id not in self.student_widgets:
            print(f"Dashboard: Adding new student chart for {student_id}")
            student_widget = StudentChartWidget(student_id)
            
            # Add a separator line
            separator = QFrame()
            separator.setFrameShape(QFrame.Shape.HLine) # Horizontal line
            separator.setFrameShadow(QFrame.Shadow.Sunken)
            separator.setFixedHeight(2) # Make it thin

            self.students_layout.addWidget(student_widget) # Add the chart widget
            self.students_layout.addWidget(separator)      # Add the separator below it

            self.student_widgets[student_id] = student_widget
            return student_widget
        else:
            return self.student_widgets[student_id]

    @Slot(str, float, str)
    def update_student_status(self, student_id, concentration, status_text):
        student_widget = self._get_or_create_student_widget(student_id)
        student_widget.update_concentration(concentration)
        # We ignore status_text for now in the chart view

    @Slot(str, str)
    def update_student_trigger(self, student_id, word):
        student_widget = self._get_or_create_student_widget(student_id)
        student_widget.update_trigger_word(word)

# --- OSC Server Runner ---
def run_osc_server(ip, port, handler, server_name="Student UI"):
    """Generic OSC server runner."""
    disp = dispatcher.Dispatcher()
    
    # Dynamically map based on handler type
    if isinstance(handler, OSCHandler): # Student UI Listener
        disp.map(OSC_TOGGLE_ADDRESS, handler.handle_message)
        disp.map(OSC_DISPLAY_ADDRESS, handler.handle_message)
        disp.map(OSC_SET_CONCENTRATION, handler.handle_message)
    elif isinstance(handler, DashboardOSCHandler): # Dashboard Listener
         # Use specific handlers from DashboardOSCHandler
         disp.map(DASHBOARD_STATUS_ADDRESS, handler.handle_status, needs_reply_address=False)
         disp.map(DASHBOARD_TRIGGER_ADDRESS, handler.handle_trigger, needs_reply_address=False)
    else:
        print(f"Error: Unknown handler type for OSC server {server_name}")
        return

    try:
        server = osc_server.ThreadingOSCUDPServer((ip, port), disp)
        print(f"{server_name} OSC Server serving on {server.server_address}")
        server.serve_forever()
    except OSError as e:
         print(f"Error starting {server_name} OSC Server on {ip}:{port} - {e}")
    except Exception as e:
        print(f"An unexpected error occurred in {server_name} OSC server: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Set background for plots (optional, 'w' is white)
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k') # 'k' is black

    app = QApplication(sys.argv)

    # --- Student UI Setup ---
    student_osc_handler = OSCHandler()
    # Optional: Start Student UI listener thread if needed for /setConcentration
    # student_osc_thread = threading.Thread(
    #    target=run_osc_server,
    #    args=(OSC_IP, OSC_PORT, student_osc_handler, "Student UI Listener"),
    #    daemon=True)
    # student_osc_thread.start()

    student_window = MainWindow(student_osc_handler, STUDENT_ID)

    # --- Dashboard Setup ---
    dashboard_osc_handler = DashboardOSCHandler()
    dashboard_osc_thread = threading.Thread(
        target=run_osc_server,
        args=(DASHBOARD_IP, DASHBOARD_PORT, dashboard_osc_handler, "Dashboard Listener"),
        daemon=True)
    dashboard_osc_thread.start()

    if not dashboard_osc_thread.is_alive():
         print("Dashboard OSC server thread failed to start.")
         # sys.exit(1) # Optionally exit

    dashboard_window = DashboardWindow(dashboard_osc_handler)

    # --- Show Windows ---
    student_window.show()
    dashboard_window.show() # Show the dashboard window

    sys.exit(app.exec()) 