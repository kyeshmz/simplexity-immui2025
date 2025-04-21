from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout,
    QLabel, QStatusBar, QSlider, QFormLayout, QApplication
)
from PySide6.QtCore import Qt, Slot, QPoint, QTimer, QEvent
from PySide6.QtGui import QTextCharFormat, QColor, QTextDocument, QTextCursor, QCursor

from annotation_popup import AnnotationPopup
from hover_text_edit import HoverTextEdit
from config import (
    DASHBOARD_TRIGGER_ADDRESS, DASHBOARD_STATUS_ADDRESS,
    ACTIVE_HIGHLIGHT_COLOR, SAMPLE_PARAGRAPH, PARAGRAPH_HOVER_CONTENT
)

from pythonosc.udp_client import SimpleUDPClient


class MainWindow(QMainWindow):
    def __init__(self, osc_handler, student_id, dashboard_ip, dashboard_port):
        super().__init__()
        self.osc_handler = osc_handler
        self.student_id = student_id
        self.setWindowTitle(f"Student Annotation UI - {self.student_id}")
        self.setGeometry(100, 100, 700, 600)

        try:
            self.osc_client = SimpleUDPClient(dashboard_ip, dashboard_port)
            print(f"Student UI OSC Client configured to send to {dashboard_ip}:{dashboard_port}")
        except Exception as e:
            print(f"Error creating OSC Client: {e}. Dashboard messages will not be sent.")
            self.osc_client = None
        print(f"[Confirmation] self.osc_client is {'NOT None' if self.osc_client else 'None'}")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.annotation_popup = AnnotationPopup()

        self.text_edit = HoverTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setText(SAMPLE_PARAGRAPH)
        font = self.text_edit.font()
        font.setPointSize(font.pointSize() + 2)
        self.text_edit.setFont(font)
        self.main_layout.addWidget(self.text_edit)

        self.binary_concentration_state = 1

        self.control_layout = QHBoxLayout()

        self.toggle_button = QPushButton("Toggle Annotations (OFF)")
        self.control_layout.addWidget(self.toggle_button)
        self.control_layout.addStretch(1)

        self.concentration_layout = QFormLayout()
        self.concentration_slider = QSlider(Qt.Orientation.Horizontal)
        self.concentration_slider.setRange(0, 100)
        self.concentration_slider.setValue(50)
        self.concentration_label = QLabel(f"{self.concentration_slider.value()/100.0:.2f}")
        self.concentration_layout.addRow(QLabel("Set Concentration:"), self.concentration_slider)
        self.concentration_layout.addRow(QLabel("Current Value:"), self.concentration_label)
        self.control_layout.addLayout(self.concentration_layout)

        self.test_trigger_button = QPushButton("Send Test Trigger")
        self.control_layout.addWidget(self.test_trigger_button)
        self.control_layout.addStretch(1)

        self.main_layout.addLayout(self.control_layout)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Annotations OFF")

        self.osc_handler.annotation_state_changed.connect(self.update_annotation_state)
        self.toggle_button.clicked.connect(self.update_annotation_state_from_button)
        self.text_edit.word_hovered.connect(self.handle_hover_event)
        self.text_edit.viewport().installEventFilter(self)
        self.osc_handler.display_annotation_requested.connect(self.show_popup_via_osc)
        self.concentration_slider.valueChanged.connect(self.update_concentration_display)
        self.concentration_slider.sliderReleased.connect(self.send_concentration_update)
        self.osc_handler.concentration_received.connect(self.set_concentration_from_osc)
        self.test_trigger_button.clicked.connect(self.send_test_trigger)

        self.update_annotation_state(self.osc_handler.show_annotations)
        self.send_concentration_update()

        self.last_hovered_paragraph_index = None

    @Slot()
    def update_annotation_state_from_button(self):
        new_state = not self.osc_handler.show_annotations
        self.osc_handler.show_annotations = new_state
        print(f"Button toggled global annotations to: {new_state}")
        self.update_annotation_state(new_state)

    @Slot(bool)
    def update_annotation_state(self, show):
        self.toggle_button.setText(f"Toggle Annotations ({'ON' if show else 'OFF'})")
        self.status_bar.showMessage(f"Annotations {'ON' if show else 'OFF'}")
        if not show:
            self.annotation_popup.hide()

    def get_visible_paragraph_index(self, cursor):
        block = self.text_edit.document().begin()
        index = 1
        while block.isValid():
            if block == cursor.block():
                return index
            if block.text().strip():
                index += 1
            block = block.next()
        return None

    def get_block_by_visible_index(self, target_index):
        block = self.text_edit.document().begin()
        index = 1
        while block.isValid():
            if block.text().strip():
                if index == target_index:
                    return block
                index += 1
            block = block.next()
        return QTextCursor().block()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseMove and obj is self.text_edit.viewport():
            if not self.osc_handler.show_annotations:
                return False
            cursor = self.text_edit.cursorForPosition(event.pos())
            paragraph_index = self.get_visible_paragraph_index(cursor)
            if paragraph_index and paragraph_index != self.last_hovered_paragraph_index:
                self.clear_paragraph_highlight()
                self.last_hovered_paragraph_index = paragraph_index
                self.highlight_paragraph(paragraph_index)
                self.show_binary_annotation(self.text_edit.mapToGlobal(event.pos()), paragraph_index)
            return False
        elif event.type() == QEvent.Leave and obj is self.text_edit.viewport():
            self.annotation_popup.hide()
            self.clear_paragraph_highlight()
            self.last_hovered_paragraph_index = None
            return True
        return super().eventFilter(obj, event)

    def show_binary_annotation(self, global_pos, paragraph_index=None):
        if not self.osc_handler.show_annotations:
            return

        if paragraph_index is None:
            cursor = self.text_edit.cursorForPosition(self.text_edit.mapFromGlobal(global_pos))
            paragraph_index = cursor.blockNumber() + 1

        state = self.binary_concentration_state
        content = PARAGRAPH_HOVER_CONTENT.get(paragraph_index, {}).get(state)

        if not content:
            self.annotation_popup.hide()
            return

        self.annotation_popup.setContent(content["text"], content["image"])
        self.annotation_popup.showAt(global_pos + QPoint(15, 15))

    @Slot(str, QPoint)
    def handle_hover_event(self, word, global_pos):
        self.show_binary_annotation(global_pos)

    @Slot(str)
    def show_popup_via_osc(self, word):
        if not self.osc_handler.show_annotations or not word:
            self.annotation_popup.hide()
            return

        doc = self.text_edit.document()
        cursor = doc.find(word, 0, QTextDocument.FindFlag.FindWholeWords)

        if cursor.isNull():
            print(f"Word '{word}' not found in the current text.")
            self.annotation_popup.hide()
            return

        word_rect = self.text_edit.cursorRect(cursor)
        global_pos_at_word = self.text_edit.mapToGlobal(word_rect.topLeft())
        popup_pos = global_pos_at_word + QPoint(0, word_rect.height() + 5)

        self.show_binary_annotation(popup_pos, cursor.blockNumber() + 1)

        if self.osc_client:
            try:
                self.osc_client.send_message(DASHBOARD_TRIGGER_ADDRESS, [self.student_id, word])
                print(f"Sent trigger: {word} for {self.student_id}")
            except Exception as e:
                print(f"Error sending trigger OSC: {e}")

    @Slot(int)
    def update_concentration_display(self, value):
        float_value = value / 100.0
        self.concentration_label.setText(f"{float_value:.2f}")

    @Slot()
    def send_concentration_update(self):
        float_value = self.concentration_slider.value() / 100.0
        print(f"Sending concentration update for {self.student_id}: {float_value}")
        if self.osc_client:
            try:
                self.osc_client.send_message(DASHBOARD_STATUS_ADDRESS, [self.student_id, float_value, ""])
            except Exception as e:
                print(f"Error sending concentration OSC: {e}")

    @Slot(float)
    def set_concentration_from_osc(self, value):
        slider_value = int(value * 100)
        self.concentration_slider.setValue(slider_value)
        self.concentration_label.setText(f"{value:.2f}")
        self.binary_concentration_state = 1 if value >= 0.5 else 2

    @Slot()
    def send_test_trigger(self):
        print("[Test Button] Trigger sending is disabled for now.")

    def highlight_paragraph(self, paragraph_index):
        block = self.get_block_by_visible_index(paragraph_index)
        cursor = QTextCursor(block)
        cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
        fmt = QTextCharFormat()
        fmt.setBackground(QColor("yellow"))
        fmt.setForeground(QColor("black"))
        cursor.setCharFormat(fmt)
        self.last_highlighted_block = paragraph_index

    def clear_paragraph_highlight(self):
        if hasattr(self, 'last_highlighted_block') and self.last_highlighted_block:
            block = self.get_block_by_visible_index(self.last_highlighted_block)
            cursor = QTextCursor(block)
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
            cursor.setCharFormat(QTextCharFormat())
            self.last_highlighted_block = None