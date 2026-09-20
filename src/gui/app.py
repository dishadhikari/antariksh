import sys

import cv2

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFrame,
)

from src.gui.runtime_worker import RuntimeWorker


class ExperimentGUI(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "SIH - BAS Human Activity Recognition"
        )

        self.resize(1200, 750)

        self.running = False
        self.worker = None

        self.setup_ui()

    # ---------------------------------
    # UI
    # ---------------------------------
    def add_event(self, message):
        self.event_log.append(message)

    def setup_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        # Title
        title = QLabel(
            "BAS Experiment Monitor"
        )

        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)

        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)

        main_layout.addWidget(title)

        # ---------------------------------
        # Main content
        # ---------------------------------

        content_layout = QHBoxLayout()

        # Camera panel
        camera_frame = QFrame()
        camera_frame.setFrameShape(QFrame.Box)

        camera_layout = QVBoxLayout()
        camera_frame.setLayout(camera_layout)

        self.camera_label = QLabel(
            "CAMERA FEED"
        )

        self.camera_label.setAlignment(
            Qt.AlignCenter
        )

        self.camera_label.setMinimumSize(
            700,
            450
        )

        camera_layout.addWidget(
            self.camera_label
        )

        content_layout.addWidget(
            camera_frame,
            stretch=2
        )

        # Information panel
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.Box)

        info_layout = QVBoxLayout()
        info_frame.setLayout(info_layout)

        self.expected_label = QLabel(
            "Expected Step: --"
        )

        self.detected_label = QLabel(
            "Detected Step: --"
        )

        self.confidence_label = QLabel(
            "Confidence: --"
        )

        self.status_label = QLabel(
            "Status: READY"
        )

        for label in [
            self.expected_label,
            self.detected_label,
            self.confidence_label,
            self.status_label,
        ]:
            label.setFont(
                QFont("Arial", 13)
            )

            info_layout.addWidget(label)

        info_layout.addSpacing(20)

        info_layout.addWidget(
            QLabel("Event Log")
        )

        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)

        info_layout.addWidget(
            self.event_log
        )

        content_layout.addWidget(
            info_frame,
            stretch=1
        )

        main_layout.addLayout(
            content_layout
        )

        # ---------------------------------
        # Controls
        # ---------------------------------

        button_layout = QHBoxLayout()

        self.start_button = QPushButton(
            "Start"
        )

        self.stop_button = QPushButton(
            "Stop"
        )

        self.reset_button = QPushButton(
            "Reset"
        )

        button_layout.addWidget(
            self.start_button
        )

        button_layout.addWidget(
            self.stop_button
        )

        button_layout.addWidget(
            self.reset_button
        )

        main_layout.addLayout(
            button_layout
        )

        self.start_button.clicked.connect(
            self.start_camera
        )

        self.stop_button.clicked.connect(
            self.stop_camera
        )

        self.reset_button.clicked.connect(
            self.reset_gui
        )

        self.stop_button.setEnabled(False)

    # ---------------------------------
    # Start runtime
    # ---------------------------------

    def start_camera(self):

        if self.running:
            return

        self.worker = RuntimeWorker(
    camera_source=0,
    detector_model="yolo11n.pt",
    har_model=None,
    voice_model=None,
    device="cpu",
    process_every=3,
)

        self.worker.frame_ready.connect(
            self.display_frame
        )

        self.worker.status_update.connect(
            self.update_status
        )

        self.worker.event_detected.connect(
            self.add_event
        )

        self.worker.error.connect(
            self.runtime_error
        )

        self.worker.finished.connect(
            self.worker_finished
        )

        self.running = True

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.status_label.setText(
            "Status: STARTING..."
        )

        self.event_log.append(
            "Runtime started."
        )

        self.worker.start()

    # ---------------------------------
    # Stop runtime
    # ---------------------------------

    def stop_camera(self):

        if not self.running:
            return

        self.running = False

        if self.worker is not None:

            self.worker.stop()

            self.worker = None

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.status_label.setText(
            "Status: STOPPED"
        )

        self.event_log.append(
            "Runtime stopped."
        )

    # ---------------------------------
    # Display camera frame
    # ---------------------------------

    def display_frame(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        height, width, channels = (
            rgb_frame.shape
        )

        bytes_per_line = (
            channels * width
        )

        image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(
            image
        )

        pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.camera_label.setPixmap(
            pixmap
        )

    # ---------------------------------
    # Runtime status
    # ---------------------------------

    def update_status(
        self,
        expected,
        detected,
        confidence,
        status,
    ):

        self.expected_label.setText(
            f"Expected Step: {expected}"
        )

        self.detected_label.setText(
            f"Detected Step: {detected}"
        )

        self.confidence_label.setText(
            f"Confidence: {confidence:.2f}"
        )

        self.status_label.setText(
            f"Status: {status}"
        )

    # ---------------------------------
    # Runtime error
    # ---------------------------------

    def runtime_error(self, message):

        self.event_log.append(
            f"ERROR: {message}"
        )

        self.status_label.setText(
            "Status: ERROR"
        )

    # ---------------------------------
    # Worker finished
    # ---------------------------------

    def worker_finished(self):

        self.running = False

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    # ---------------------------------
    # Reset
    # ---------------------------------

    def reset_gui(self):

        if self.running:
            self.stop_camera()

        self.expected_label.setText(
            "Expected Step: --"
        )

        self.detected_label.setText(
            "Detected Step: --"
        )

        self.confidence_label.setText(
            "Confidence: --"
        )

        self.status_label.setText(
            "Status: READY"
        )

        self.event_log.clear()

        self.camera_label.clear()

        self.camera_label.setText(
            "CAMERA FEED"
        )

    # ---------------------------------
    # Close application
    # ---------------------------------

    def closeEvent(self, event):

        if self.worker is not None:
            self.worker.stop()
            self.worker = None

        self.running = False

        event.accept()


def main():

    app = QApplication(sys.argv)

    window = ExperimentGUI()
    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()