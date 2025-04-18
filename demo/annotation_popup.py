import os
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

class AnnotationPopup(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Window flags to make it look like a tooltip/popup
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        # Layout and content widgets
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(100, 100) # Min size for image area
        self.text_label = QLabel()
        self.text_label.setWordWrap(True)
        self.text_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.text_label)

        # Styling (optional, customize as needed)
        self.setStyleSheet("""
            AnnotationPopup {
                background-color: #ffffff;
                border: 1px solid #aaaaaa;
                border-radius: 4px;
            }
            QLabel {
                color: #333333;
                background-color: transparent; /* Ensure labels don't obscure frame background */
            }
        """)

    def setContent(self, text, image_path):
        # Set Text
        self.text_label.setText(text)
        self.text_label.setVisible(bool(text))

        # Set Image
        pixmap = None
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                 print(f"Error loading image: {image_path}")
                 pixmap = None # Failed to load
        elif image_path:
             print(f"Image file not found: {image_path}")

        if pixmap:
            # Simple scaling logic (adjust as needed)
            max_width = 200
            if pixmap.width() > max_width:
                pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.image_label.setVisible(True)
            self.image_label.setFixedSize(pixmap.size()) # Adjust label size to pixmap
        else:
            self.image_label.clear()
            self.image_label.setVisible(False)
            self.image_label.setFixedSize(0,0) # Collapse if no image

        # Adjust the popup size to fit content
        self.adjustSize()

    def showAt(self, pos: QPoint):
        # Ensure position is reasonable (within screen bounds if possible)
        # Basic check - more sophisticated checks might be needed
        screen_geo = QApplication.primaryScreen().availableGeometry()
        popup_rect = self.geometry()
        popup_rect.moveTopLeft(pos)

        # Adjust if going off-screen (simple adjustment)
        if popup_rect.right() > screen_geo.right():
            pos.setX(screen_geo.right() - popup_rect.width() - 5)
        if popup_rect.bottom() > screen_geo.bottom():
            pos.setY(screen_geo.bottom() - popup_rect.height() - 5)
        if popup_rect.left() < screen_geo.left():
            pos.setX(screen_geo.left() + 5)
        if popup_rect.top() < screen_geo.top():
            pos.setY(screen_geo.top() + 5)

        self.move(pos)
        self.show() 