import os
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QApplication
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap

class AnnotationPopup(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(100, 100)

        self.text_label = QLabel()
        self.text_label.setWordWrap(True)
        self.text_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.text_label)

        self.setStyleSheet("""
            AnnotationPopup {
                background-color: #ffffff;
                border: 1px solid #aaaaaa;
                border-radius: 4px;
            }
            QLabel {
                color: #333333;
                background-color: transparent;
            }
        """)

    def setContent(self, text, image_path):
        if not self.isVisible():
            self.hide()  # Make sure we're not half-drawn

        self.text_label.setText(text or "")
        self.text_label.setVisible(bool(text))

        pixmap = None
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                print(f"[AnnotationPopup] Failed to load image: {image_path}")
                pixmap = None
        elif image_path:
            print(f"[AnnotationPopup] Image not found: {image_path}")

        if pixmap:
            max_width = 200
            if pixmap.width() > max_width:
                pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.image_label.setVisible(True)
            self.image_label.setFixedSize(pixmap.size())
        else:
            self.image_label.clear()
            self.image_label.setVisible(False)
            self.image_label.setFixedSize(0, 0)

        self.adjustSize()

    def showAt(self, pos: QPoint):
        if not self.isWidgetType() or not self.text_label or not self.image_label:
            print("[AnnotationPopup] Popup in invalid state — skipping showAt()")
            return

        screen_geo = QApplication.primaryScreen().availableGeometry()
        popup_rect = self.geometry()
        popup_rect.moveTopLeft(pos)

        if popup_rect.right() > screen_geo.right():
            pos.setX(screen_geo.right() - popup_rect.width() - 5)
        if popup_rect.bottom() > screen_geo.bottom():
            pos.setY(screen_geo.bottom() - popup_rect.height() - 5)
        if popup_rect.left() < screen_geo.left():
            pos.setX(screen_geo.left() + 5)
        if popup_rect.top() < screen_geo.top():
            pos.setY(screen_geo.top() + 5)

        try:
            self.move(pos)
            if not self.isVisible():
                self.show()
        except RuntimeError as e:
            print(f"[AnnotationPopup] Error showing popup: {e}")
