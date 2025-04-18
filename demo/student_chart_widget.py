from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

import pyqtgraph as pg
from pyqtgraph import PlotWidget, mkPen

class StudentChartWidget(QWidget):
    def __init__(self, student_id, parent=None):
        super().__init__(parent)
        self.student_id = student_id
        self.max_data_points = 100  # Max points to show on chart

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
        self.plot_widget.setYRange(-0.1, 1.1)  # Adjusted Y range for binary
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        # Set explicit ticks for binary values
        self.plot_widget.getAxis('left').setTicks([[(0, '0'), (1, '1')]])
        self.plot_widget.getAxis('bottom').setTicks([])  # Hide bottom ticks for simplicity
        
        # Data storage for the plot
        self.time_data = list(range(self.max_data_points))  # X-axis (indices)
        self.concentration_data = [0.0] * self.max_data_points  # Y-axis

        # Create plot curve item
        self.plot_curve = self.plot_widget.plot(self.time_data, self.concentration_data, pen=mkPen('b', width=2))  # Blue line

        # Layout
        self.layout.addWidget(self.id_label)
        self.layout.addWidget(self.plot_widget)
        self.layout.addWidget(self.trigger_label)

        # Set size policy to expand horizontally but be fixed vertically
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def update_concentration(self, concentration):
        # Convert to binary
        binary_concentration = 1.0 if concentration >= 0.5 else 0.0

        # Shift data left, add new *binary* point at the end
        self.concentration_data = self.concentration_data[1:] + [binary_concentration]
        # Update the plot curve data
        self.plot_curve.setData(self.time_data, self.concentration_data)

        # Update color based on *binary* value
        # Get color based on binary value (green for concentrated, orange for not)
        color = QColor("lightgreen") if binary_concentration else QColor("orange")
        self.id_label.setStyleSheet(f"background-color: {color.name()}; padding: 2px; border-radius: 3px;")

    def update_trigger_word(self, word):
        self.trigger_label.setText(f"Last Trigger: {word}") 