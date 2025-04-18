from PySide6.QtWidgets import QTextEdit
from PySide6.QtCore import Signal, QPoint, Qt
from PySide6.QtGui import QTextCursor, QMouseEvent

from config import ANNOTATION_WORDS

class HoverTextEdit(QTextEdit):
    # Emits the word being hovered over (or empty string) and the global mouse position
    word_hovered = Signal(str, QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.last_hovered_word = ""

    def mouseMoveEvent(self, event: QMouseEvent):
        cursor = self.cursorForPosition(event.pos())
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        selected_word = cursor.selectedText().strip('.,?!;:"\'()')

        current_word = ""
        if selected_word and selected_word in ANNOTATION_WORDS:
            current_word = selected_word

        # Emit word and global position if word changes
        if current_word != self.last_hovered_word:
            self.word_hovered.emit(current_word, event.globalPos())
            self.last_hovered_word = current_word
        # Also emit if mouse moves significantly over the *same* word, so popup follows
        elif current_word:
            # You might want to add a distance threshold check here
            # to avoid emitting too frequently
            self.word_hovered.emit(current_word, event.globalPos())

        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self.last_hovered_word != "":
            self.last_hovered_word = ""
            self.word_hovered.emit("", QPoint()) # Emit empty word, position irrelevant
        super().leaveEvent(event) 