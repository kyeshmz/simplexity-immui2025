from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout,
    QLabel, QStatusBar, QSlider, QFormLayout, QApplication, QGraphicsOpacityEffect
)
from PySide6.QtCore import Qt, Slot, QPoint, QTimer
from PySide6.QtGui import QTextCharFormat, QColor, QTextDocument, QTextCursor, QPixmap, QPainter, QBrush

from annotation_popup import AnnotationPopup
from hover_text_edit import HoverTextEdit
from config import (
    DASHBOARD_TRIGGER_ADDRESS, DASHBOARD_STATUS_ADDRESS,
    ACTIVE_HIGHLIGHT_COLOR, SAMPLE_PARAGRAPH, PARAGRAPH_HOVER_CONTENT
)

from pythonosc.udp_client import SimpleUDPClient
from eyeGesturesWrapper import EyeGestures
from collections import deque


class MainWindow(QMainWindow):
    def __init__(self, osc_handler, student_id, dashboard_ip, dashboard_port):
        super().__init__()
        self.osc_handler = osc_handler
        self.student_id = student_id
        self.setWindowTitle(f"Student Annotation UI - {self.student_id}")
        self.setGeometry(100, 100, 1200, 800)

        try:
            self.osc_client = SimpleUDPClient(dashboard_ip, dashboard_port)
            print(f"Student UI OSC Client configured to send to {dashboard_ip}:{dashboard_port}")
        except Exception as e:
            print(f"Error creating OSC Client: {e}")
            self.osc_client = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.annotation_popup = AnnotationPopup()

        self.text_edit = HoverTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setText(SAMPLE_PARAGRAPH)
        font = self.text_edit.font()
        font.setPointSize(20)
        self.text_edit.setFont(font)
        self.main_layout.addWidget(self.text_edit)

        # # --- Heatmap overlay setup ---
        # self.heatmap_overlay = QLabel(self.text_edit.viewport())
        # self.heatmap_overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        # self.heatmap_overlay.lower()  # put it behind text if needed

        self.gaze_trail = deque(maxlen=10)

        self.binary_concentration_state = 1

        self.control_layout = QHBoxLayout()
        self.toggle_button = QPushButton("Toggle Annotations (OFF)")
        self.control_layout.addWidget(self.toggle_button)
        self.control_layout.addStretch(1)

        self.concentration_layout = QFormLayout()
        self.concentration_slider = QSlider(Qt.Orientation.Horizontal)
        self.concentration_slider.setRange(0, 100)
        self.concentration_slider.setValue(50)
        self.concentration_label = QLabel(f"{self.concentration_slider.value() / 100.0:.2f}")
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

        self.eye_gestures = EyeGestures()
        self.gaze_timer = QTimer()
        self.gaze_timer.timeout.connect(self.check_gaze_position)
        self.gaze_timer.start(16)

        self.last_hovered_paragraph_index = None

        self.osc_handler.annotation_state_changed.connect(self.update_annotation_state)
        self.toggle_button.clicked.connect(self.update_annotation_state_from_button)
        self.text_edit.viewport().installEventFilter(self)
        self.osc_handler.display_annotation_requested.connect(self.show_popup_via_osc)
        self.concentration_slider.valueChanged.connect(self.update_concentration_display)
        self.concentration_slider.sliderReleased.connect(self.send_concentration_update)
        self.osc_handler.concentration_received.connect(self.set_concentration_from_osc)
        self.test_trigger_button.clicked.connect(self.send_test_trigger)

        self.update_annotation_state(self.osc_handler.show_annotations)
        self.send_concentration_update()

        self.smoothed_x = 0.5
        self.smoothed_y = 0.5
        self.smoothing_alpha = 0.18

        self.show()
        # QTimer.singleShot(0, self.resize_heatmap_overlay)



    @Slot()
    def update_annotation_state_from_button(self):
        new_state = not self.osc_handler.show_annotations
        self.osc_handler.show_annotations = new_state
        self.update_annotation_state(new_state)

    @Slot(bool)
    def update_annotation_state(self, show):
        self.toggle_button.setText(f"Toggle Annotations ({'ON' if show else 'OFF'})")
        self.status_bar.showMessage(f"Annotations {'ON' if show else 'OFF'}")
        if not show:
            self.annotation_popup.hide()

    def get_visible_paragraph_index(self, cursor):
        if not cursor.isNull():
            block = cursor.block()
            block_index = 1
            current = self.text_edit.document().begin()
            while current.isValid():
                if current == block:
                    return block_index
                if current.text().strip():
                    block_index += 1
                current = current.next()
        return None

    def check_gaze_position(self):
        if not self.osc_handler.show_annotations:
            return

        # norm_x, norm_y = self.eye_gestures.process_frame(
        #     screen_width=self.text_edit.width(),
        #     screen_height=self.text_edit.height()
        # )
        # if norm_x is None or norm_y is None:
        #     return

        norms = self.eye_gestures.process_frame(...)
        if norms is not None:
            norm_x, norm_y = norms
            self.last_valid_x = norm_x
            self.last_valid_y = norm_y
        else:
            norm_x, norm_y = self.last_valid_x, self.last_valid_y

        # Smooth the gaze to reduce jitter
        self.smoothed_x = (
                self.smoothing_alpha * norm_x + (1 - self.smoothing_alpha) * self.smoothed_x
        )
        self.smoothed_y = (
                self.smoothing_alpha * norm_y + (1 - self.smoothing_alpha) * self.smoothed_y
        )

        widget_x = int(self.smoothed_x * self.text_edit.viewport().width())
        widget_y = int(self.smoothed_y * self.text_edit.viewport().height())

        widget_pos = QPoint(widget_x, widget_y)

        # # --- Heatmap drawing ---
        # self.fade_heatmap()
        # self.draw_heat_point(widget_x, widget_y)

        # --- Paragraph highlight and popup logic ---
        cursor = self.text_edit.cursorForPosition(widget_pos)

        if not cursor.isNull():
            paragraph_index = self.get_visible_paragraph_index(cursor)
            if paragraph_index and paragraph_index != self.last_hovered_paragraph_index:
                self.clear_paragraph_highlight()
                self.last_hovered_paragraph_index = paragraph_index
                self.highlight_paragraph(paragraph_index)
                self.show_binary_annotation(self.text_edit.viewport().mapToGlobal(widget_pos), paragraph_index)

        # Clear old dot if over limit
        if len(self.gaze_trail) >= 5:
            old_dot = self.gaze_trail.popleft()
            old_dot.deleteLater()

        # Create the new dot
        dot = QLabel(self.text_edit.viewport())
        size = 20
        dot.setFixedSize(size, size)
        dot.setStyleSheet(f"background-color: rgba(0, 128, 255, 180); border-radius: {size // 2}px;")
        dot.move(widget_x - size // 2, widget_y - size // 2)
        dot.show()

        self.gaze_trail.append(dot)

        # Apply progressively fading opacity
        for i, dot in enumerate(reversed(self.gaze_trail)):
            opacity_effect = QGraphicsOpacityEffect()
            alpha = max(0.2, 1.0 - (i / len(self.gaze_trail)))  # Fades from 1.0 down to ~0.2
            opacity_effect.setOpacity(alpha)
            dot.setGraphicsEffect(opacity_effect)

    # def fade_heatmap(self, fade_strength=10):
    #     fade_painter = QPainter(self.heatmap_pixmap)
    #     fade_painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
    #     # fade_color = QColor(0, 0, 0, 255 - fade_strength)  # keep 90% of the previous frame
    #     fade_color = QColor(0, 0, 0, 255)
    #     fade_painter.fillRect(self.heatmap_pixmap.rect(), fade_color)
    #     fade_painter.end()
    #     self.heatmap_overlay.setPixmap(self.heatmap_pixmap)
    #
    # def draw_heat_point(self, x, y, radius=15, alpha=80):
    #     painter = QPainter(self.heatmap_pixmap)
    #     color = QColor(0, 128, 255, alpha)
    #     painter.setBrush(QBrush(color))
    #     painter.setPen(Qt.NoPen)
    #     painter.drawEllipse(QPoint(x, y), radius, radius)
    #     painter.end()
    #     self.heatmap_overlay.setPixmap(self.heatmap_pixmap)

    def show_binary_annotation(self, global_pos, paragraph_index=None):
        if not self.osc_handler.show_annotations:
            return

        if paragraph_index is None:
            cursor = self.text_edit.cursorForPosition(self.text_edit.viewport().mapFromGlobal(global_pos))
            paragraph_index = cursor.blockNumber() + 1

        state = self.binary_concentration_state
        content = PARAGRAPH_HOVER_CONTENT.get(paragraph_index, {}).get(state)

        if not content:
            self.annotation_popup.hide()
            return

        self.annotation_popup.setContent(content["text"], content["image"])
        self.annotation_popup.showAt(global_pos + QPoint(15, 15))

    @Slot(str)
    def show_popup_via_osc(self, word):
        if not self.osc_handler.show_annotations or not word:
            self.annotation_popup.hide()
            return

        doc = self.text_edit.document()
        cursor = doc.find(word, 0, QTextDocument.FindWholeWords)
        if cursor.isNull():
            self.annotation_popup.hide()
            return

        word_rect = self.text_edit.cursorRect(cursor)
        popup_pos = self.text_edit.viewport().mapToGlobal(word_rect.topLeft())
        self.show_binary_annotation(popup_pos, cursor.blockNumber() + 1)

        if self.osc_client:
            try:
                self.osc_client.send_message(DASHBOARD_TRIGGER_ADDRESS, [self.student_id, word])
            except Exception as e:
                print(f"Error sending trigger OSC: {e}")

    @Slot(int)
    def update_concentration_display(self, value):
        float_value = value / 100.0
        self.concentration_label.setText(f"{float_value:.2f}")

    @Slot()
    def send_concentration_update(self):
        value = self.concentration_slider.value() / 100.0
        if self.osc_client:
            try:
                self.osc_client.send_message(DASHBOARD_STATUS_ADDRESS, [self.student_id, value, ""])
            except Exception as e:
                print(f"Error sending concentration OSC: {e}")

    @Slot(float)
    def set_concentration_from_osc(self, value):
        self.concentration_slider.setValue(int(value * 100))
        self.concentration_label.setText(f"{value:.2f}")
        self.binary_concentration_state = 1 if value >= 0.5 else 2

    @Slot()
    def send_test_trigger(self):
        print("[Test Button] Trigger sending is disabled for now.")

    # def resize_heatmap_overlay(self):
    #     self.heatmap_overlay.resize(self.text_edit.viewport().size())
    #     self.heatmap_pixmap = QPixmap(self.heatmap_overlay.size())
    #     self.heatmap_pixmap.fill(Qt.transparent)
    #     self.heatmap_overlay.setPixmap(self.heatmap_pixmap)
    #
    # def resizeEvent(self, event):
    #     super().resizeEvent(event)
    #     self.resize_heatmap_overlay()

    def highlight_paragraph(self, paragraph_index):
        block = self.get_block_by_visible_index(paragraph_index)
        cursor = QTextCursor(block)
        cursor.select(QTextCursor.BlockUnderCursor)
        fmt = QTextCharFormat()
        fmt.setBackground(QColor("yellow"))
        fmt.setForeground(QColor("black"))
        cursor.setCharFormat(fmt)
        self.last_highlighted_block = paragraph_index

    def clear_paragraph_highlight(self):
        if hasattr(self, 'last_highlighted_block') and self.last_highlighted_block:
            block = self.get_block_by_visible_index(self.last_highlighted_block)
            cursor = QTextCursor(block)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.setCharFormat(QTextCharFormat())
            self.last_highlighted_block = None

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
