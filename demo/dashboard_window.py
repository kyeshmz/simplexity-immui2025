from PySide6.QtWidgets import QMainWindow, QScrollArea, QWidget, QVBoxLayout, QFrame
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor

from student_chart_widget import StudentChartWidget


class DashboardWindow(QMainWindow):
    # Define color mapping for BINARY concentration levels
    CONCENTRATION_COLORS = {
        0: QColor("orange"),  # Not concentrated
        1: QColor("lightgreen")  # Concentrated
    }

    @staticmethod
    def get_concentration_color(binary_concentration):  # Input is now 0 or 1
        return DashboardWindow.CONCENTRATION_COLORS.get(int(binary_concentration), QColor("white"))  # Default white

    def __init__(self, osc_handler):
        super().__init__()
        self.osc_handler = osc_handler
        self.setWindowTitle("Student Dashboard")
        self.setGeometry(900, 100, 400, 700)  # Positioned to the right of main window

        # Scrollable Area Setup
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # No horizontal scroll
        self.setCentralWidget(self.scroll_area)

        # Container widget inside scroll area
        self.container_widget = QWidget()
        self.scroll_area.setWidget(self.container_widget)

        # Layout for the container (holds student chart widgets)
        self.students_layout = QVBoxLayout(self.container_widget)
        self.students_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # Add new students at the top

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
            separator.setFrameShape(QFrame.Shape.HLine)  # Horizontal line
            separator.setFrameShadow(QFrame.Shadow.Sunken)
            separator.setFixedHeight(2)  # Make it thin

            self.students_layout.addWidget(student_widget)  # Add the chart widget
            self.students_layout.addWidget(separator)  # Add the separator below it

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