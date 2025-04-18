from PySide6.QtCore import QObject, Signal

class DashboardOSCHandler(QObject):
    student_status_updated = Signal(str, float, str)
    student_word_triggered = Signal(str, str)

    def __init__(self):
        super().__init__()
        
    def handle_status(self, address, student_id, concentration, *args):
        status_text = args[0] if args else ""
        # print(f"Dashboard Rx Status: ID:{student_id} Conc:{concentration} Status:'{status_text}'") # Less verbose
        try:
            self.student_status_updated.emit(str(student_id), float(concentration), status_text)
        except ValueError:
            print(f"Error: Bad concentration value '{concentration}' for student {student_id}.")
        except Exception as e:
            print(f"Error processing status message: {e}")
            
    def handle_trigger(self, address, student_id, word, *args):
        # print(f"Dashboard Rx Trigger: ID:{student_id} Word:'{word}'") # Less verbose
        try:
            self.student_word_triggered.emit(str(student_id), str(word))
        except Exception as e:
            print(f"Error processing trigger message: {e}") 