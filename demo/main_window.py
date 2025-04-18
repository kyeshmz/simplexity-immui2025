from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout,
    QLabel, QStatusBar, QSlider, QFormLayout, QApplication
)
from PySide6.QtCore import Qt, Slot, QPoint, QTimer
from PySide6.QtGui import QTextCharFormat, QColor, QTextDocument, QTextCursor

from annotation_popup import AnnotationPopup
from hover_text_edit import HoverTextEdit
from config import (
    ANNOTATION_WORDS, DASHBOARD_TRIGGER_ADDRESS, DASHBOARD_STATUS_ADDRESS, 
    ACTIVE_HIGHLIGHT_COLOR, SAMPLE_PARAGRAPH
)

from pythonosc.udp_client import SimpleUDPClient

class MainWindow(QMainWindow):
    def __init__(self, osc_handler, student_id, dashboard_ip, dashboard_port):
        super().__init__()
        self.osc_handler = osc_handler
        self.student_id = student_id
        self.setWindowTitle(f"Student Annotation UI - {self.student_id}")
        self.setGeometry(100, 100, 700, 600) # Increased height for concentration controls

        # OSC Client to send data to Dashboard
        try:
            self.osc_client = SimpleUDPClient(dashboard_ip, dashboard_port)
            print(f"Student UI OSC Client configured to send to {dashboard_ip}:{dashboard_port}")
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
        self.highlight_format.setBackground(QColor(ACTIVE_HIGHLIGHT_COLOR))

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

        # Apply highlight to the word
        self.apply_highlight(cursor)

        # Update popup content and show it
        text = annotation_data.get('text', '')
        image_path = annotation_data.get('image')
        self.annotation_popup.setContent(text, image_path)
        self.annotation_popup.showAt(popup_pos)

        # Send trigger message to dashboard
        if self.osc_client:
            try:
                self.osc_client.send_message(DASHBOARD_TRIGGER_ADDRESS, [self.student_id, word])
                print(f"Sent trigger: {word} for {self.student_id}")
            except Exception as e:
                print(f"Error sending trigger OSC: {e}")

    @Slot(bool)
    def update_annotation_state(self, show):
        """Updates UI elements based on the global annotation state (button, status bar)."""
        self.toggle_button.setText(f"Toggle Annotations ({'ON' if show else 'OFF'})")
        self.status_bar.showMessage(f"Annotations {'ON' if show else 'OFF'}")
        if not show:
            self.annotation_popup.hide()
            self.clear_active_highlight()

    # --- Concentration Methods ---
    @Slot(int)
    def update_concentration_display(self, value):
        """Updates the label next to the slider."""
        float_value = value / 100.0
        self.concentration_label.setText(f"{float_value:.2f}")

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
                    if hasattr(widget, 'update_student_trigger'):
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
        self.active_highlight_cursor = None
        self.active_highlight_original_format = None

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