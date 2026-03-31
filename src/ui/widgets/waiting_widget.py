"""Waiting widget displayed when no annotation job is active."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QScrollArea, QFrame, QPushButton,
)
from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QFont


class SpinnerWidget(QWidget):
    """Animated spinning arc indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 80)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._rotate)
        self._timer.start(16)  # ~60 FPS

    def _rotate(self):
        self._angle = (self._angle + 4) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        size = min(self.width(), self.height())
        rect = QRectF(
            (self.width() - size) / 2 + 6,
            (self.height() - size) / 2 + 6,
            size - 12,
            size - 12,
        )

        # Draw background circle (subtle)
        bg_pen = QPen(QColor(255, 255, 255, 30), 4)
        bg_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(bg_pen)
        painter.drawEllipse(rect)

        # Draw spinning arc
        pen = QPen(QColor(88, 166, 255), 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, int(self._angle * 16), int(270 * 16))

        painter.end()

    def start(self):
        self._timer.start(16)

    def stop(self):
        self._timer.stop()


class PulsingDot(QWidget):
    """Small pulsing dot to indicate active polling."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self._opacity = 0.3
        self._growing = True
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._pulse)
        self._timer.start(50)

    def _pulse(self):
        if self._growing:
            self._opacity += 0.02
            if self._opacity >= 1.0:
                self._opacity = 1.0
                self._growing = False
        else:
            self._opacity -= 0.02
            if self._opacity <= 0.3:
                self._opacity = 0.3
                self._growing = True
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(76, 175, 80)
        color.setAlphaF(self._opacity)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(1, 1, 10, 10)
        painter.end()

    def start(self):
        self._timer.start(50)

    def stop(self):
        self._timer.stop()


class _FileProgressRow(QWidget):
    """One row: filename label + progress bar."""

    _BAR_STYLE = """
        QProgressBar {
            background-color: #313244;
            border: none;
            border-radius: 4px;
            height: 8px;
            text-align: right;
            color: #cdd6f4;
            font-size: 10px;
        }
        QProgressBar::chunk {
            background-color: #58a6ff;
            border-radius: 4px;
        }
    """

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(12)

        name_lbl = QLabel(label)
        name_lbl.setFixedWidth(160)
        name_lbl.setFont(QFont("", 10))
        name_lbl.setStyleSheet("color: #a6adc8; background: transparent;")
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(name_lbl)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)   # use permilles for smooth display
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%p%")
        self._bar.setStyleSheet(self._BAR_STYLE)
        self._bar.setFixedHeight(18)
        layout.addWidget(self._bar, 1)

        self._size_lbl = QLabel("—")
        self._size_lbl.setFixedWidth(70)
        self._size_lbl.setFont(QFont("", 9))
        self._size_lbl.setStyleSheet("color: #585b70; background: transparent;")
        self._size_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._size_lbl)

    def update_progress(self, done: int, total: int) -> None:
        if total > 0:
            permille = int(done * 1000 / total)
            self._bar.setValue(permille)
            pct = done * 100 // total
            mb_done = done / 1_000_000
            mb_total = total / 1_000_000
            self._bar.setFormat(f"{pct}%")
            self._size_lbl.setText(f"{mb_done:.1f}/{mb_total:.1f} Mo")
        else:
            # Total unknown — pulse the bar
            self._bar.setRange(0, 0)
            if done > 0:
                self._size_lbl.setText(f"{done / 1_000_000:.1f} Mo")


class WaitingWidget(QWidget):
    """Clean idle screen shown while waiting for a RabbitMQ job."""

    skip_requested = pyqtSignal()
    load_from_nas_requested = pyqtSignal()
    load_from_spool_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # label -> _FileProgressRow
        self._rows: dict[str, _FileProgressRow] = {}
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("""
            WaitingWidget {
                background-color: #1e1e2e;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(24)

        layout.addStretch()

        # Spinner
        spinner_container = QWidget()
        spinner_layout = QVBoxLayout(spinner_container)
        spinner_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinner = SpinnerWidget()
        spinner_layout.addWidget(self.spinner, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(spinner_container)

        # Title
        title = QLabel("En attente d'un job")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("", 20, QFont.Weight.DemiBold))
        title.setStyleSheet("color: #cdd6f4; background: transparent;")
        layout.addWidget(title)

        # Subtitle (status)
        subtitle = QLabel("Connexion au serveur RabbitMQ...")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setFont(QFont("", 12))
        subtitle.setStyleSheet("color: #6c7086; background: transparent;")
        self.subtitle_label = subtitle
        layout.addWidget(subtitle)

        # Dot + status text
        status_container = QWidget()
        status_container.setStyleSheet("background: transparent;")
        status_layout = QVBoxLayout(status_container)
        status_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.setSpacing(8)

        dot_row = QWidget()
        dot_row.setStyleSheet("background: transparent;")
        dot_layout = QHBoxLayout(dot_row)
        dot_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dot_layout.setSpacing(8)
        dot_layout.setContentsMargins(0, 0, 0, 0)

        self.pulsing_dot = PulsingDot()
        dot_layout.addWidget(self.pulsing_dot)

        self.status_label = QLabel("Polling actif")
        self.status_label.setFont(QFont("", 10))
        self.status_label.setStyleSheet("color: #a6adc8; background: transparent;")
        dot_layout.addWidget(self.status_label)

        status_layout.addWidget(dot_row, alignment=Qt.AlignmentFlag.AlignCenter)

        self.queue_label = QLabel("")
        self.queue_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.queue_label.setFont(QFont("", 10))
        self.queue_label.setStyleSheet("color: #585b70; background: transparent;")
        status_layout.addWidget(self.queue_label)

        layout.addWidget(status_container)

        # File progress area (hidden until download starts)
        self._progress_frame = QFrame()
        self._progress_frame.setStyleSheet("""
            QFrame {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 8px;
            }
        """)
        self._progress_frame.setMinimumWidth(480)
        self._progress_frame.setMaximumWidth(640)
        self._progress_frame.hide()

        frame_layout = QVBoxLayout(self._progress_frame)
        frame_layout.setContentsMargins(16, 12, 16, 12)
        frame_layout.setSpacing(4)

        dl_header = QLabel("Téléchargement en cours")
        dl_header.setFont(QFont("", 11, QFont.Weight.DemiBold))
        dl_header.setStyleSheet("color: #cdd6f4; background: transparent; border: none;")
        frame_layout.addWidget(dl_header)

        # Container for progress rows
        self._rows_container = QWidget()
        self._rows_container.setStyleSheet("background: transparent;")
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 8, 0, 0)
        self._rows_layout.setSpacing(2)
        frame_layout.addWidget(self._rows_container)

        layout.addWidget(self._progress_frame, alignment=Qt.AlignmentFlag.AlignCenter)

        # Skip button — visible uniquement pendant le téléchargement
        self._skip_btn = QPushButton("⏭  Passer ce job")
        self._skip_btn.setFixedSize(180, 38)
        self._skip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._skip_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #e64553;
                border-color: #e64553;
                color: white;
            }
            QPushButton:pressed {
                background-color: #c0392b;
            }
        """)
        self._skip_btn.clicked.connect(self.skip_requested)
        self._skip_btn.hide()
        layout.addWidget(self._skip_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # NAS browse button — always visible
        self._nas_btn = QPushButton("📂  Charger depuis le NAS")
        self._nas_btn.setFixedSize(220, 38)
        self._nas_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nas_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e3a5f;
                color: #58a6ff;
                border: 1px solid #2a5298;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2a5298;
                border-color: #58a6ff;
                color: white;
            }
            QPushButton:pressed {
                background-color: #1a3a7a;
            }
        """)
        self._nas_btn.clicked.connect(self.load_from_nas_requested)
        layout.addWidget(self._nas_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # SPOOL button — récupérer un scénario depuis le serveur SPOOL
        self._spool_btn = QPushButton("📥  Charger depuis le SPOOL")
        self._spool_btn.setFixedSize(240, 38)
        self._spool_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._spool_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a1f3d;
                color: #cba6f7;
                border: 1px solid #6c5a9e;
                border-radius: 8px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #6c5a9e;
                border-color: #cba6f7;
                color: white;
            }
            QPushButton:pressed {
                background-color: #1a1030;
            }
        """)
        self._spool_btn.clicked.connect(self.load_from_spool_requested)
        layout.addWidget(self._spool_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_status(self, text: str) -> None:
        self.subtitle_label.setText(text)

    def set_queue_info(self, text: str) -> None:
        self.queue_label.setText(text)

    def start_animation(self) -> None:
        self.spinner.start()
        self.pulsing_dot.start()

    def stop_animation(self) -> None:
        self.spinner.stop()
        self.pulsing_dot.stop()

    def reset_file_progress(self) -> None:
        """Clear all file progress rows (call before starting a new download)."""
        for row in self._rows.values():
            row.deleteLater()
        self._rows.clear()
        self._progress_frame.hide()

    def update_file_progress(self, label: str, done: int, total: int) -> None:
        """Create or update the progress bar for *label*.

        Safe to call from any thread via a Qt QueuedConnection.
        """
        if label not in self._rows:
            row = _FileProgressRow(label)
            self._rows[label] = row
            self._rows_layout.addWidget(row)
            self._progress_frame.show()

        self._rows[label].update_progress(done, total)

    def show_skip_button(self) -> None:
        """Affiche le bouton 'Passer ce job' (pendant un téléchargement ou une attente)."""
        self._skip_btn.show()

    def hide_skip_button(self) -> None:
        """Cache le bouton 'Passer ce job'."""
        self._skip_btn.hide()
