"""PySide6 GUI for XTTSv2 voice cloning."""

import os
import tempfile

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import sounddevice as sd
from scipy.io.wavfile import write

from audio_recorder import MicrophoneRecorder
from voice_cloner import VoiceCloner


class SynthesisWorker(QThread):
    finished = Signal(object, int)
    progress = Signal(int)

    def __init__(self, cloner: VoiceCloner, text: str, sample: str) -> None:
        super().__init__()
        self.cloner = cloner
        self.text = text
        self.sample = sample

    def run(self) -> None:
        self.progress.emit(0)
        audio, sr = self.cloner.synthesize(self.text, self.sample)
        self.progress.emit(100)
        self.finished.emit(audio, sr)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("XTTSv2 Voice Cloner")

        self.cloner = VoiceCloner()
        self.recorder = MicrophoneRecorder()
        self.sample_path: str | None = None
        self.output_audio = None
        self.output_sr: int | None = None

        layout = QVBoxLayout()

        self.sample_label = QLabel("No sample selected")
        layout.addWidget(self.sample_label)

        btn_layout = QHBoxLayout()
        self.select_btn = QPushButton("Select Sample")
        self.select_btn.clicked.connect(self.select_sample)
        btn_layout.addWidget(self.select_btn)
        self.record_btn = QPushButton("Record")
        self.record_btn.clicked.connect(self.toggle_recording)
        btn_layout.addWidget(self.record_btn)
        layout.addLayout(btn_layout)

        self.text_edit = QTextEdit()
        layout.addWidget(self.text_edit)

        self.generate_btn = QPushButton("Generate Speech")
        self.generate_btn.clicked.connect(self.generate)
        layout.addWidget(self.generate_btn)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        play_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        play_layout.addWidget(self.play_btn)
        self.save_btn = QPushButton("Save As...")
        self.save_btn.clicked.connect(self.save_audio)
        self.save_btn.setEnabled(False)
        play_layout.addWidget(self.save_btn)
        layout.addLayout(play_layout)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def select_sample(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select voice sample", filter="Audio files (*.wav *.mp3)")
        if path:
            self.sample_path = path
            self.sample_label.setText(os.path.basename(path))

    def toggle_recording(self) -> None:
        if not self.recorder.recording:
            self.recorder.start_recording()
            self.record_btn.setText("Stop")
            self.sample_label.setText("Recording...")
        else:
            data = self.recorder.stop_recording()
            if data.size == 0:
                QMessageBox.warning(self, "Recording failed", "No audio captured")
                return
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            self.recorder.save_recording(tmp.name, data)
            self.sample_path = tmp.name
            self.sample_label.setText(f"Recorded: {os.path.basename(tmp.name)}")
            self.record_btn.setText("Record")

    def generate(self) -> None:
        if not self.sample_path:
            QMessageBox.warning(self, "No sample", "Select or record a sample first")
            return
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No text", "Enter text to synthesize")
            return

        self.generate_btn.setEnabled(False)
        self.progress.setValue(0)
        self.worker = SynthesisWorker(self.cloner, text, self.sample_path)
        self.worker.finished.connect(self.synthesis_done)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.start()

    def synthesis_done(self, audio, sr) -> None:
        self.output_audio = audio
        self.output_sr = sr
        self.generate_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

    def play_audio(self) -> None:
        if self.output_audio is not None and self.output_sr is not None:
            sd.play(self.output_audio, self.output_sr)

    def save_audio(self) -> None:
        if self.output_audio is None or self.output_sr is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save audio", filter="WAV files (*.wav)")
        if path:
            write(path, self.output_sr, self.output_audio)


def main() -> None:
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
