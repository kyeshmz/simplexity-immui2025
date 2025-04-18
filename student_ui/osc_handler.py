from PySide6.QtCore import Signal, QObject

# Import config - adjust path if necessary
from utils.config import OSC_TOGGLE_ADDRESS, OSC_DISPLAY_ADDRESS, OSC_SET_CONCENTRATION

class OSCHandler(QObject):
    """Handles OSC messages specifically for the Student UI."""
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
                # Ensure arg exists and is convertible to int then bool
                new_state = bool(int(args[0]))
                if new_state != self.show_annotations:
                    self.show_annotations = new_state
                    self.annotation_state_changed.emit(self.show_annotations)
                    print(f"Student UI Annotation state changed via OSC to: {self.show_annotations}")
            except (IndexError, ValueError, TypeError) as e:
                print(f"Error processing {OSC_TOGGLE_ADDRESS} message ({args}): {e}. Expected 0 or 1.")

        elif address == OSC_DISPLAY_ADDRESS:
            try:
                # Ensure arg exists and is convertible to str
                word_to_display = str(args[0]) if args else ""
                if word_to_display: # Only emit if a word was actually sent
                    print(f"Student UI OSC request to display annotation for: '{word_to_display}'")
                    self.display_annotation_requested.emit(word_to_display)
                else:
                    print(f"Warning: Received empty {OSC_DISPLAY_ADDRESS} message.")
            except IndexError:
                 print(f"Error processing {OSC_DISPLAY_ADDRESS} message: Missing argument.")
            except Exception as e: # Catch other potential errors during string conversion
                 print(f"Error processing {OSC_DISPLAY_ADDRESS} message ({args}): {e}")

        elif address == OSC_SET_CONCENTRATION: # Handle external concentration input
            try:
                # Ensure arg exists and is convertible to float
                conc_float = float(args[0])
                # Clamp value between 0.0 and 1.0
                conc_float = max(0.0, min(1.0, conc_float))
                self.concentration_received.emit(conc_float)
                print(f"Student UI Received concentration: {conc_float:.2f}")
            except (IndexError, ValueError, TypeError) as e:
                print(f"Error processing {OSC_SET_CONCENTRATION} message ({args}): {e}. Expected a float.") 