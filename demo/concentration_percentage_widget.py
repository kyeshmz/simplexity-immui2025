from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QFont

class ConcentrationPercentageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.concentrated_count = 0
        self.total_count = 0
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # Title label
        self.title_label = QLabel("Class Concentration")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(14)
        self.title_label.setFont(font)
        
        # Percentage label
        self.percentage_label = QLabel("0%")
        self.percentage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(24)
        self.percentage_label.setFont(font)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimumHeight(20)
        
        # Count label
        self.count_label = QLabel("0/0 students concentrated")
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add widgets to layout
        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.percentage_label)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.count_label)
        
        # Set size policies
        self.setMinimumHeight(150)
        
        print("Concentration Percentage Widget initialized")
        
    def update_concentration(self, concentrated_count, total_count):
        print(f"Updating concentration percentage: {concentrated_count}/{total_count} students concentrated")
        
        self.concentrated_count = concentrated_count
        self.total_count = total_count
        
        if total_count > 0:
            percentage = (concentrated_count / total_count) * 100
        else:
            percentage = 0
            
        print(f"New concentration percentage: {percentage:.1f}%")
            
        self.percentage_label.setText(f"{int(percentage)}%")
        self.progress_bar.setValue(int(percentage))
        self.count_label.setText(f"{concentrated_count}/{total_count} students concentrated")
        
        # Update progress bar color based on percentage
        style = ""
        if percentage < 30:
            style = "QProgressBar::chunk { background-color: red; }"
        elif percentage < 70:
            style = "QProgressBar::chunk { background-color: orange; }"
        else:
            style = "QProgressBar::chunk { background-color: green; }"
            
        self.progress_bar.setStyleSheet(style) 