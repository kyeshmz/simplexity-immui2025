import sys
import threading
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt, Signal, Slot, QObject
from PySide6.QtGui import QColor

from pythonosc import dispatcher
from pythonosc import osc_server

# --- Configuration ---
DASHBOARD_OSC_IP = "127.0.0.1"  # Listen on localhost
DASHBOARD_OSC_PORT = 5006       # Port for dashboard to listen on
STATUS_ADDRESS = "/student/status"  # Address for concentration/status updates
TRIGGER_ADDRESS = "/student/trigger" # Address for word trigger updates

# --- OSC Communication Handler (for Dashboard) ---
class DashboardOSCHandler(QObject):
    # Signals to update the UI safely from the OSC thread
    # Arguments: student_id (str), concentration (float), optional_status_text (str)
    student_status_updated = Signal(str, float, str)
    # Arguments: student_id (str), triggered_word (str)
    student_word_triggered = Signal(str, str)

    def __init__(self):
        super().__init__()

    def handle_status(self, address, student_id, concentration, *args):
        """Handles incoming /student/status messages."""
        status_text = args[0] if args else ""
        print(f"Dashboard Received Status: {address} ID:{student_id} Conc:{concentration} Status:'{status_text}'")
        try:
            conc_float = float(concentration)
            self.student_status_updated.emit(str(student_id), conc_float, status_text)
        except ValueError:
            print(f"Error: Could not convert concentration '{concentration}' to float for student {student_id}.")
        except Exception as e:
             print(f"Error processing status message: {e}")

    def handle_trigger(self, address, student_id, word, *args):
        """Handles incoming /student/trigger messages."""
        print(f"Dashboard Received Trigger: {address} ID:{student_id} Word:'{word}'")
        try:
            self.student_word_triggered.emit(str(student_id), str(word))
        except Exception as e:
             print(f"Error processing trigger message: {e}")


# --- Dashboard Application Window ---
class DashboardWindow(QMainWindow):
    # Define color mapping for concentration levels (example)
    CONCENTRATION_COLORS = {
        (0.0, 0.3): QColor("red"),
        (0.3, 0.7): QColor("orange"),
        (0.7, 1.0): QColor("lightgreen")
    }

    def __init__(self, osc_handler):
        super().__init__()
        self.osc_handler = osc_handler
        self.setWindowTitle("Student Status Dashboard")
        self.setGeometry(200, 200, 600, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Table to display student data
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(3) # Columns: ID, Concentration, Last Triggered Word
        self.table_widget.setHorizontalHeaderLabels(["Student ID", "Concentration", "Last Triggered Word"])
        self.table_widget.verticalHeader().setVisible(False) # Hide row numbers
        # Make columns resize nicely
        self.table_widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.table_widget.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_widget.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        self.layout.addWidget(self.table_widget)

        # Internal data storage: {student_id: {"row": row_index, "concentration": 0.0, "last_word": ""}}
        self.student_data = {}

        # Connect signals from OSC handler to slots
        self.osc_handler.student_status_updated.connect(self.update_student_status)
        self.osc_handler.student_word_triggered.connect(self.update_student_trigger)


    def get_concentration_color(self, concentration):
        """Returns a QColor based on the concentration value."""
        for (low, high), color in self.CONCENTRATION_COLORS.items():
            # Handle edge case for 1.0 belonging to the top range
            if low <= concentration <= high:
                 # Special case for exact upper bound if it's not the max possible value (1.0)
                if concentration == high and high != 1.0:
                    continue # Look for the next range if available
                return color
        return QColor("white") # Default color if no range matches

    @Slot(str, float, str)
    def update_student_status(self, student_id, concentration, status_text):
        """Updates the concentration and status text for a student in the table."""
        if student_id not in self.student_data:
            # Add new student row
            row_position = self.table_widget.rowCount()
            self.table_widget.insertRow(row_position)
            self.student_data[student_id] = {"row": row_position, "concentration": concentration, "last_word": ""}
            # Add ID item (non-editable)
            id_item = QTableWidgetItem(student_id)
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_widget.setItem(row_position, 0, id_item)
            # Add concentration item
            conc_item = QTableWidgetItem(f"{concentration:.2f}")
            conc_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            conc_item.setFlags(conc_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_widget.setItem(row_position, 1, conc_item)
             # Add trigger item
            trigger_item = QTableWidgetItem("") # Initially empty
            trigger_item.setFlags(trigger_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_widget.setItem(row_position, 2, trigger_item)
        else:
            # Update existing student
            row_position = self.student_data[student_id]["row"]
            self.student_data[student_id]["concentration"] = concentration
            conc_item = self.table_widget.item(row_position, 1)
            if not conc_item: # Should not happen if row was created correctly
                 conc_item = QTableWidgetItem()
                 conc_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                 conc_item.setFlags(conc_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                 self.table_widget.setItem(row_position, 1, conc_item)
            conc_item.setText(f"{concentration:.2f}")


        # Update background color based on concentration
        color = self.get_concentration_color(concentration)
        conc_item_to_color = self.table_widget.item(row_position, 1)
        if conc_item_to_color:
            conc_item_to_color.setBackground(color)

        # Optionally display status_text somewhere (e.g., tooltip or separate column)
        # For now, just update internal data
        self.student_data[student_id]["status_text"] = status_text # If needed later


    @Slot(str, str)
    def update_student_trigger(self, student_id, word):
        """Updates the last triggered word for a student in the table."""
        if student_id not in self.student_data:
            # Student triggered a word before sending status? Add them.
            print(f"Trigger received for unknown student {student_id}. Adding row.")
            # Add with default concentration 0.0 temporarily
            self.update_student_status(student_id, 0.0, "")
            # Now update the word

        # Update existing student
        row_position = self.student_data[student_id]["row"]
        self.student_data[student_id]["last_word"] = word

        trigger_item = self.table_widget.item(row_position, 2)
        if not trigger_item: # Should not happen
            trigger_item = QTableWidgetItem()
            trigger_item.setFlags(trigger_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table_widget.setItem(row_position, 2, trigger_item)
        trigger_item.setText(word)


# --- OSC Server Thread ---
def run_osc_server(ip, port, handler):
    disp = dispatcher.Dispatcher()
    # Map the addresses to the handler methods
    # Use default handlers that catch all arguments after the required ones
    disp.map(STATUS_ADDRESS, handler.handle_status, needs_reply_address=False)
    disp.map(TRIGGER_ADDRESS, handler.handle_trigger, needs_reply_address=False)

    try:
        server = osc_server.ThreadingOSCUDPServer((ip, port), disp)
        print(f"Dashboard OSC Server serving on {server.server_address}")
        server.serve_forever()
    except OSError as e:
         print(f"Error starting Dashboard OSC Server on {ip}:{port} - {e}")
         print("Please check if the port is already in use.")
    except Exception as e:
        print(f"An unexpected error occurred in Dashboard OSC server: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create OSC handler (lives in main thread but handles signals)
    osc_handler = DashboardOSCHandler()

    # Start OSC server in a separate thread
    osc_thread = threading.Thread(target=run_osc_server, args=(DASHBOARD_OSC_IP, DASHBOARD_OSC_PORT, osc_handler), daemon=True)
    osc_thread.start()

    if not osc_thread.is_alive() and threading.active_count() <= 2:
         print("Dashboard OSC server thread failed to start. Exiting.")
         # sys.exit(1) # Optionally exit

    # Create and show the main window
    dashboard_window = DashboardWindow(osc_handler)
    dashboard_window.show()

    sys.exit(app.exec()) 