from PySide6.QtCore import QObject, Signal, Slot

from config import OSC_TOGGLE_ADDRESS, OSC_DISPLAY_ADDRESS, OSC_SET_CONCENTRATION

class OSCHandler(QObject):
    """Handles OSC messages for the student UI."""
    annotation_state_changed = Signal(bool)
    display_annotation_requested = Signal(str)
    concentration_received = Signal(float)

    def __init__(self):
        super().__init__()
        self.show_annotations = False  # Initial state is OFF

    def handle_message(self, address, *args):
        # Called by OSC server dispatcher
        if address == OSC_TOGGLE_ADDRESS:
            if args and len(args) > 0:
                # Parse value from OSC message
                if isinstance(args[0], bool):
                    new_state = args[0]
                elif isinstance(args[0], int):
                    new_state = bool(args[0])
                elif isinstance(args[0], str):
                    new_state = args[0].lower() in ('true', 'on', '1', 'yes')
                else:
                    print(f"Invalid toggle value: {args[0]}")
                    return
                
                # Update state and emit signal
                if new_state != self.show_annotations:
                    self.show_annotations = new_state
                    print(f"OSC: Toggled annotations to {new_state}")
                    self.annotation_state_changed.emit(new_state)
            else:
                # Toggle current state if no value provided
                self.show_annotations = not self.show_annotations
                print(f"OSC: Toggled annotations to {self.show_annotations}")
                self.annotation_state_changed.emit(self.show_annotations)
                
        elif address == OSC_DISPLAY_ADDRESS:
            if args and len(args) > 0 and self.show_annotations:
                word = str(args[0])
                print(f"OSC: Requested display of annotation for: {word}")
                self.display_annotation_requested.emit(word)
                
        elif address == OSC_SET_CONCENTRATION:
            if args and len(args) > 0:
                try:
                    concentration = float(args[0])
                    # Clamp to 0.0-1.0 range
                    concentration = max(0.0, min(1.0, concentration))
                    print(f"OSC: Received concentration value: {concentration}")
                    self.concentration_received.emit(concentration)
                except (ValueError, TypeError) as e:
                    print(f"Error parsing concentration value '{args[0]}': {e}") 