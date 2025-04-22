from PySide6.QtWidgets import QMainWindow, QScrollArea, QWidget, QVBoxLayout, QFrame, QHBoxLayout, QGridLayout
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor

from student_indicator_widget import StudentIndicatorWidget
from concentration_percentage_widget import ConcentrationPercentageWidget


class DashboardWindow(QMainWindow):
    # Define color mapping for BINARY concentration levels
    CONCENTRATION_COLORS = {
        0: QColor("red"),       # Not concentrated
        1: QColor("green")      # Concentrated
    }

    @staticmethod
    def get_concentration_color(binary_concentration):  # Input is now 0 or 1
        return DashboardWindow.CONCENTRATION_COLORS.get(int(binary_concentration), QColor("white"))  # Default white

    def __init__(self, osc_handler):
        super().__init__()
        self.osc_handler = osc_handler
        self.setWindowTitle("Student Dashboard")
        self.setGeometry(900, 100, 500, 700)  # Positioned to the right of main window

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Concentration percentage widget at the top
        self.percentage_widget = ConcentrationPercentageWidget()
        self.main_layout.addWidget(self.percentage_widget)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setFixedHeight(2)
        self.main_layout.addWidget(separator)
        
        # Scrollable Area for Student Indicators
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.main_layout.addWidget(self.scroll_area, 1)  # Give scroll area more space
        
        # Container widget inside scroll area
        self.container_widget = QWidget()
        self.scroll_area.setWidget(self.container_widget)
        
        # Layout for the container (holds student indicators)
        self.students_layout = QVBoxLayout(self.container_widget)
        self.students_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Internal data storage
        self.student_widgets = {}
        
        # Connect signals from OSC handler
        print("Connecting OSC handler signals to dashboard slots")
        self.osc_handler.student_status_updated.connect(self.update_student_status)
        self.osc_handler.student_word_triggered.connect(self.update_student_trigger)
        print("Signal connections established")

    def _get_or_create_student_widget(self, student_id):
        """Finds existing widget or creates and adds a new one."""
        if student_id not in self.student_widgets:
            print(f"Dashboard: Creating new student indicator widget for {student_id}")
            student_widget = StudentIndicatorWidget(student_id)
            
            # Add a separator line
            separator = QFrame()
            separator.setFrameShape(QFrame.Shape.HLine)
            separator.setFrameShadow(QFrame.Shadow.Sunken)
            separator.setFixedHeight(1)
            
            self.students_layout.addWidget(student_widget)
            self.students_layout.addWidget(separator)
            
            self.student_widgets[student_id] = student_widget
            
            self._update_concentration_percentage()
            return student_widget
        else:
            print(f"Dashboard: Found existing widget for {student_id}")
            return self.student_widgets[student_id]
            
    def _update_concentration_percentage(self):
        """Update the concentration percentage display based on all students."""
        concentrated_count = 0
        total_count = len(self.student_widgets)
        
        print(f"Updating concentration percentage from {total_count} student widgets")
        
        for student_id, widget in self.student_widgets.items():
            if widget.is_concentrated:
                concentrated_count += 1
                print(f"  Student {student_id} is concentrated")
            else:
                print(f"  Student {student_id} is distracted") 
                
        print(f"Total concentration: {concentrated_count}/{total_count}")
        self.percentage_widget.update_concentration(concentrated_count, total_count)

    @Slot(str, float, str)
    def update_student_status(self, student_id, concentration, status_text):
        print(f"Dashboard Window received update for {student_id}: concentration={concentration}, status={status_text}")
        
        # Get or create widget
        student_widget = self._get_or_create_student_widget(student_id)
        
        # Apply the concentration update to the widget
        print(f"Updating widget for {student_id} with concentration {concentration}")
        student_widget.update_concentration(concentration)
        
        # Use QApplication.processEvents to ensure UI updates immediately
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        
        # Update the percentage display
        print(f"Updating percentage display after {student_id} concentration change")
        self._update_concentration_percentage()

    @Slot(str, str)
    def update_student_trigger(self, student_id, word):
        print(f"Dashboard Window received trigger for {student_id}: word={word}")
        student_widget = self._get_or_create_student_widget(student_id)
        student_widget.update_trigger_word(word)