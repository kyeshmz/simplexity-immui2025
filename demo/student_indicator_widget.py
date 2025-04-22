from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QColor, QPainter, QBrush, QPen, QFont

class StudentIndicatorWidget(QWidget):
    def __init__(self, student_id, parent=None):
        super().__init__(parent)
        self.student_id = student_id
        self.is_concentrated = False
        self.blink_state = False
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.toggle_blink)
        self.blink_timer.start(500)  # Blink every 500ms
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        # Student ID label
        self.id_label = QLabel(f"{self.student_id}")
        font = QFont()
        font.setBold(True)
        self.id_label.setFont(font)
        self.id_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.id_label.setMinimumWidth(100)
        
        # Status label
        self.status_label = QLabel("UNKNOWN")
        font = QFont()
        font.setBold(True)
        self.status_label.setFont(font)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Last trigger word label
        self.trigger_label = QLabel("")
        self.trigger_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.layout.addWidget(self.id_label)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.trigger_label, 1)  # Give trigger label more space
        
        self.setMinimumHeight(50)
        self.setMaximumHeight(50)
        
        print(f"Created indicator widget for student: {self.student_id}")
        
        # Set initial appearance
        self._update_ui_elements()
    
    def toggle_blink(self):
        self.blink_state = not self.blink_state
        self.update()  # Trigger repaint
    
    def update_concentration(self, concentration):
        # Convert to binary
        prev_state = self.is_concentrated
        self.is_concentrated = concentration >= 0.5
        
        print(f"Student {self.student_id} concentration updated: {concentration:.2f} → binary: {1 if self.is_concentrated else 0}")
        
        if prev_state != self.is_concentrated:
            print(f"Student {self.student_id} state changed from {'concentrated' if prev_state else 'distracted'} "
                  f"to {'concentrated' if self.is_concentrated else 'distracted'}")
            
            # Force immediate update of visual state
            self._update_ui_elements()
            
            # Make sure the repaint happens right away 
            self.repaint()
        else:
            # Even if the binary state didn't change, still update the UI
            self._update_ui_elements()
            self.update()  # Schedule repaint for next event loop iteration
    
    def _update_ui_elements(self):
        """Update UI elements based on concentration state"""
        # Update status text - use only two states
        if self.is_concentrated:
            self.status_label.setText("ENGAGED")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.status_label.setText("DISTRACTED")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        
        # Update label styles
        if self.is_concentrated:
            self.id_label.setStyleSheet("background-color: green; color: white; padding: 4px; border-radius: 3px;")
        else:
            self.id_label.setStyleSheet("background-color: red; color: white; padding: 4px; border-radius: 3px;")
        
        # Force a repaint for the blinking circle
        self.update()
    
    def update_trigger_word(self, word):
        print(f"Student {self.student_id} trigger word updated: {word}")
        self.trigger_label.setText(f"Last: {word}")
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw the status circle
        circle_size = min(self.height() - 10, 30)
        
        # Set color based on concentration (green if concentrated, red if distracted)
        if self.is_concentrated:
            color = QColor("green")
        else:
            color = QColor("red")
            
        # Reduce opacity when blinking is off
        if not self.blink_state:
            color.setAlpha(100)
            
        # Draw circle
        painter.setPen(QPen(color.darker(), 2))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(10, (self.height() - circle_size) // 2, 
                          circle_size, circle_size) 