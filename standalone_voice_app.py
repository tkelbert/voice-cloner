#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import soundfile as sf
import librosa
import torch
import wave
import tempfile
import pickle
import json
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                           QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                           QSlider, QProgressBar, QMessageBox, QGroupBox,
                           QTextEdit, QTabWidget, QSplitter, QComboBox,
                           QListWidget, QDialog, QInputDialog, QLineEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QFont
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

try:
    import pyaudio
except ImportError:
    print("PyAudio not found. Recording functionality will be disabled.")
    print("Please install with: pip install pyaudio")
    # Create a dummy pyaudio to prevent crashes
    class DummyPyAudio:
        def __init__(self):
            pass
        def open(self, *args, **kwargs):
            raise Exception("PyAudio not installed")
    pyaudio = type('DummyModule', (), {'PyAudio': DummyPyAudio, 'paInt16': 16})

# Simplified voice synthesis model
class VoiceSynthesisModel:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.sample_rate = 22050
        print(f"Voice model initialized on {device}")
        
        # Placeholder model parameters
        self.encoder_params = torch.randn(64, 32).to(device)
        self.decoder_params = torch.randn(32, 16).to(device)
        self.vocoder_params = torch.randn(16, 8).to(device)
        
    def extract_speaker_embedding(self, audio_path):
        """Extract speaker embedding from audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract features (MFCC for demonstration)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # Normalize
            mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
            
            # Convert to tensor
            mfccs_tensor = torch.FloatTensor(mfccs).to(self.device)
            
            # Simulate encoder output
            embedding = torch.matmul(torch.mean(mfccs_tensor, dim=1), self.encoder_params)
            
            # Also return the audio for visualization
            return {
                'embedding': embedding,
                'waveform': y,
                'sr': sr,
                'mfccs': mfccs
            }
        except Exception as e:
            print(f"Error extracting speaker embedding: {e}")
            return None
    
    def synthesize_speech(self, embedding, text, pitch_shift=0, speed=1.0, emotion="neutral"):
        """Synthesize speech based on speaker embedding and text"""
        try:
            # Simulate text encoding based on length
            text_encoding = torch.FloatTensor([ord(c)/128 for c in text]).to(self.device)
            
            # Pad or truncate to fixed length
            max_len = 100
            if len(text_encoding) > max_len:
                text_encoding = text_encoding[:max_len]
            else:
                text_encoding = torch.nn.functional.pad(
                    text_encoding, (0, max_len - len(text_encoding)))
            
            # Emotion factor
            emotion_factors = {
                "neutral": 1.0,
                "happy": 1.2,
                "sad": 0.8,
                "angry": 1.3,
                "surprised": 1.4
            }
            emotion_factor = emotion_factors.get(emotion, 1.0)
            
            # Generate simple audio for demonstration
            duration = len(text) / 10  # seconds
            t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
            
            # Generate a carrier wave
            carrier = np.sin(2 * np.pi * 220 * t)  # 220 Hz tone
            
            # Modulate amplitude based on text
            modulator = np.array([0.5 + 0.5 * np.sin(2 * np.pi * (ord(c) / 128) * t) 
                                for c in text[:min(10, len(text))]])
            modulator = np.mean(modulator, axis=0)
            
            y = carrier * modulator * emotion_factor
            
            # Apply pitch shift
            if pitch_shift != 0:
                y = librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=pitch_shift)
            
            # Apply speed change
            if speed != 1.0:
                y = librosa.effects.time_stretch(y, rate=speed)
            
            return y, self.sample_rate
            
        except Exception as e:
            print(f"Error synthesizing speech: {e}")
            # Return a simple beep tone as a fallback
            t = np.linspace(0, 1.0, self.sample_rate, endpoint=False)
            return np.sin(2 * np.pi * 440 * t), self.sample_rate

class AudioRecorder:
    def __init__(self, chunk=1024, channels=1, rate=22050, format=pyaudio.paInt16):
        self.CHUNK = chunk
        self.FORMAT = format
        self.CHANNELS = channels
        self.RATE = rate
        self.p = None
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.temp_file = None
    
    def start_recording(self):
        try:
            self.p = pyaudio.PyAudio()
            self.frames = []
            self.is_recording = True
            
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            print("Recording started...")
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
            raise
    
    def stop_recording(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.p:
            self.p.terminate()
            
        self.is_recording = False
        print("Recording stopped.")
        
        # Create a temporary file to store the recording
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
        wf = wave.open(self.temp_file.name, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        return self.temp_file.name
    
    def add_frame(self, data):
        if self.is_recording:
            self.frames.append(data)

class RecordingThread(QThread):
    update_signal = pyqtSignal(bytes)
    finished_signal = pyqtSignal(str)
    
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self.running = True
    
    def run(self):
        try:
            self.recorder.start_recording()
            
            while self.running and self.recorder.is_recording:
                try:
                    data = self.recorder.stream.read(self.recorder.CHUNK)
                    self.recorder.add_frame(data)
                    self.update_signal.emit(data)
                    self.msleep(10)  # Small delay
                except Exception as e:
                    print(f"Error during recording: {e}")
                    break
                    
            file_path = self.recorder.stop_recording()
            self.finished_signal.emit(file_path)
        except Exception as e:
            print(f"Recording thread error: {e}")
            self.finished_signal.emit("")
    
    def stop(self):
        self.running = False

class AnalysisThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)
    
    def __init__(self, model, audio_path):
        super().__init__()
        self.model = model
        self.audio_path = audio_path
        
    def run(self):
        # Progress updates
        for i in range(101):
            self.progress_signal.emit(i)
            self.msleep(10)
        
        # Perform analysis
        features = self.model.extract_speaker_embedding(self.audio_path)
        self.finished_signal.emit(features)

class SynthesisThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(tuple)
    
    def __init__(self, model, embedding, text, pitch_shift, speed, emotion):
        super().__init__()
        self.model = model
        self.embedding = embedding
        self.text = text
        self.pitch_shift = pitch_shift
        self.speed = speed
        self.emotion = emotion
        
    def run(self):
        # Progress updates
        for i in range(101):
            self.progress_signal.emit(i)
            self.msleep(20)
        
        # Perform synthesis
        y, sr = self.model.synthesize_speech(
            self.embedding, self.text, self.pitch_shift, self.speed, self.emotion)
        self.finished_signal.emit((y, sr))

class WaveformWidget(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.patch.set_facecolor('#2E2E2E')
        self.ax.set_facecolor('#2E2E2E')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['right'].set_color('white')
        
    def plot_waveform(self, y, sr):
        self.ax.clear()
        time = np.arange(0, len(y)) / sr
        self.ax.plot(time, y, color='#00A8E8')
        self.ax.set_title('Waveform', color='white')
        self.ax.set_xlabel('Time (s)', color='white')
        self.ax.set_ylabel('Amplitude', color='white')
        self.fig.tight_layout()
        self.draw()
        
    def plot_spectrogram(self, y, sr):
        self.ax.clear()
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        self.ax.imshow(D, extent=[0, len(y)/sr, 0, sr/2], aspect='auto', 
                      origin='lower', cmap='viridis')
        self.ax.set_title('Spectrogram', color='white')
        self.ax.set_xlabel('Time (s)', color='white')
        self.ax.set_ylabel('Frequency (Hz)', color='white')
        self.fig.tight_layout()
        self.draw()
        
    def plot_mfcc(self, mfccs):
        self.ax.clear()
        self.ax.imshow(mfccs, aspect='auto', origin='lower', cmap='viridis')
        self.ax.set_title('MFCCs', color='white')
        self.ax.set_xlabel('Time', color='white')
        self.ax.set_ylabel('MFCC Coefficients', color='white')
        self.fig.tight_layout()
        self.draw()

class VoiceHistoryItem:
    def __init__(self, text, audio, sr, timestamp=None):
        self.text = text
        self.audio = audio
        self.sr = sr
        self.timestamp = timestamp or datetime.now()
    
    def get_display_name(self):
        return f"{self.timestamp.strftime('%H:%M:%S')} - {self.text[:30]}{'...' if len(self.text) > 30 else ''}"

class ProductionVoiceCloningApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Initialize model
        self.model = VoiceSynthesisModel()
        self.speaker_embedding = None
        self.output_audio = None
        self.output_sr = None
        self.history = []
        self.current_voice_profile = "Default"
        self.voice_profiles = {
            "Default": None
        }
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Voice Synthesis Studio Pro')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E1E;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                border-radius: 4px;
                padding: 8px;
                font-size: 14px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #0098FF;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QProgressBar {
                border: 1px solid #007ACC;
                border-radius: 4px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #007ACC;
            }
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 6px;
                margin-top: 10px;
                color: white;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #444444;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #007ACC;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 4px;
            }
            QTextEdit {
                background-color: #2D2D30;
                color: white;
                border: 1px solid #3E3E42;
                border-radius: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #444444;
                border-radius: 4px;
                top: -1px;
            }
            QTabBar::tab {
                background-color: #2D2D30;
                color: white;
                border: 1px solid #444444;
                border-bottom-color: #444444;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 8ex;
                padding: 8px;
            }
            QTabBar::tab:selected {
                background-color: #007ACC;
            }
            QTabBar::tab:!selected {
                margin-top: 2px;
            }
            QListWidget {
                background-color: #2D2D30;
                color: white;
                border: 1px solid #3E3E42;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #007ACC;
            }
            QComboBox {
                background-color: #2D2D30;
                color: white;
                border: 1px solid #3E3E42;
                border-radius: 4px;
                padding: 4px;
                min-width: 6em;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: #3E3E42;
                border-left-style: solid;
            }
            QComboBox QAbstractItemView {
                background-color: #2D2D30;
                color: white;
                selection-background-color: #007ACC;
            }
        """)
        
        # Main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel (Controls)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Profile management
        profile_group = QGroupBox("Voice Profile")
        profile_layout = QVBoxLayout()
        
        profile_header = QHBoxLayout()
        profile_label = QLabel("Current Profile:")
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("Default")
        self.profile_combo.currentTextChanged.connect(self.change_profile)
        profile_header.addWidget(profile_label)
        profile_header.addWidget(self.profile_combo)
        profile_layout.addLayout(profile_header)
        
        profile_buttons = QHBoxLayout()
        self.save_profile_btn = QPushButton("Save Profile")
        self.save_profile_btn.clicked.connect(self.save_profile)
        self.save_profile_btn.setEnabled(False)
        
        self.new_profile_btn = QPushButton("New Profile")
        self.new_profile_btn.clicked.connect(self.new_profile)
        
        profile_buttons.addWidget(self.new_profile_btn)
        profile_buttons.addWidget(self.save_profile_btn)
        profile_layout.addLayout(profile_buttons)
        
        profile_group.setLayout(profile_layout)
        left_layout.addWidget(profile_group)
        
        # Source audio group
        source_group = QGroupBox("Source Voice")
        source_layout = QVBoxLayout()
        
        self.source_label = QLabel("No source audio selected")
        source_layout.addWidget(self.source_label)
        
        source_buttons = QHBoxLayout()
        self.select_source_btn = QPushButton("Select Voice Sample")
        self.select_source_btn.clicked.connect(self.select_source_audio)
        
        self.record_source_btn = QPushButton("Record Voice")
        self.record_source_btn.clicked.connect(self.record_source_audio)
        
        source_buttons.addWidget(self.select_source_btn)
        source_buttons.addWidget(self.record_source_btn)
        source_layout.addLayout(source_buttons)
        
        self.analyze_btn = QPushButton("Analyze Voice")
        self.analyze_btn.clicked.connect(self.analyze_audio)
        self.analyze_btn.setEnabled(False)
        source_layout.addWidget(self.analyze_btn)
        
        self.source_progress = QProgressBar()
        source_layout.addWidget(self.source_progress)
        
        source_group.setLayout(source_layout)
        left_layout.addWidget(source_group)
        
        # Synthesis controls group
        synthesis_group = QGroupBox("Voice Synthesis Controls")
        synthesis_layout = QVBoxLayout()
        
        # Pitch control
        pitch_layout = QVBoxLayout()
        pitch_layout.addWidget(QLabel("Pitch Adjustment:"))
        self.pitch_slider = QSlider(Qt.Horizontal)
        self.pitch_slider.setMinimum(-12)
        self.pitch_slider.setMaximum(12)
        self.pitch_slider.setValue(0)
        self.pitch_slider.setTickPosition(QSlider.TicksBelow)
        self.pitch_slider.setTickInterval(2)
        pitch_layout.addWidget(self.pitch_slider)
        
        self.pitch_label = QLabel("Current: 0 semitones")
        self.pitch_slider.valueChanged.connect(self.update_pitch_label)
        pitch_layout.addWidget(self.pitch_label)
        synthesis_layout.addLayout(pitch_layout)
        
        # Speed control
        speed_layout = QVBoxLayout()
        speed_layout.addWidget(QLabel("Speed Adjustment:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(50)
        self.speed_slider.setMaximum(150)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(10)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("Current: 1.0x")
        self.speed_slider.valueChanged.connect(self.update_speed_label)
        speed_layout.addWidget(self.speed_label)
        synthesis_layout.addLayout(speed_layout)
        
        # Emotion selection
        emotion_layout = QHBoxLayout()
        emotion_layout.addWidget(QLabel("Emotion:"))
        self.emotion_combo = QComboBox()
        self.emotion_combo.addItems(["neutral", "happy", "sad", "angry", "surprised"])
        emotion_layout.addWidget(self.emotion_combo)
        synthesis_layout.addLayout(emotion_layout)
        
        synthesis_group.setLayout(synthesis_layout)
        left_layout.addWidget(synthesis_group)
        
        left_layout.addStretch(1)
        
        # Center panel (Text input and generation)
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        
        # Text input
        text_group = QGroupBox("Text Input")
        text_layout = QVBoxLayout()
        
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Enter text for the voice to speak...")
        text_layout.addWidget(self.text_edit)
        
        # Quick text options
        quick_text_layout = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Select a preset...",
            "Hello, my name is...",
            "Welcome to my application!",
            "This is a test of the voice synthesis system.",
            "The quick brown fox jumps over the lazy dog."
        ])
        self.preset_combo.currentTextChanged.connect(self.use_preset_text)
        
        quick_text_layout.addWidget(QLabel("Quick Text:"))
        quick_text_layout.addWidget(self.preset_combo)
        text_layout.addLayout(quick_text_layout)
        
        # Generate button
        self.generate_btn = QPushButton("Generate Speech")
        self.generate_btn.clicked.connect(self.generate_speech)
        self.generate_btn.setEnabled(False)
        text_layout.addWidget(self.generate_btn)
        
        self.generate_progress = QProgressBar()
        text_layout.addWidget(self.generate_progress)
        
        # Batch processing option
        batch_layout = QHBoxLayout()
        self.batch_checkbox = QComboBox()
        self.batch_checkbox.addItems(["Single Text", "Line by Line", "Paragraph by Paragraph"])
        batch_layout.addWidget(QLabel("Processing Mode:"))
        batch_layout.addWidget(self.batch_checkbox)
        text_layout.addLayout(batch_layout)
        
        text_group.setLayout(text_layout)
        center_layout.addWidget(text_group)
        
        # Output controls
        output_group = QGroupBox("Output Controls")
        output_layout = QVBoxLayout()
        
        output_buttons = QHBoxLayout()
        self.play_output_btn = QPushButton("Play Generated Audio")
        self.play_output_btn.clicked.connect(self.play_output)
        self.play_output_btn.setEnabled(False)
        
        self.stop_output_btn = QPushButton("Stop Playback")
        self.stop_output_btn.clicked.connect(self.stop_output)
        self.stop_output_btn.setEnabled(False)
        
        self.save_output_btn = QPushButton("Save Generated Audio")
        self.save_output_btn.clicked.connect(self.save_output)
        self.save_output_btn.setEnabled(False)
        
        output_buttons.addWidget(self.play_output_btn)
        output_buttons.addWidget(self.stop_output_btn)
        output_buttons.addWidget(self.save_output_btn)
        output_layout.addLayout(output_buttons)
        
        output_group.setLayout(output_layout)
        center_layout.addWidget(output_group)
        
        # Right panel (Visualization and history)
        right_widget = QTabWidget()
        
        # Visualization tab
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        
        # Source waveform
        viz_layout.addWidget(QLabel("Source Audio:"))
        self.source_viz_tabs = QTabWidget()
        
        # Waveform tab
        waveform_widget = QWidget()
        waveform_layout = QVBoxLayout(waveform_widget)
        self.source_waveform = WaveformWidget(self, width=5, height=2)
        waveform_layout.addWidget(self.source_waveform)
        self.source_viz_tabs.addTab(waveform_widget, "Waveform")
        
        # Spectrogram tab
        spec_widget = QWidget()
        spec_layout = QVBoxLayout(spec_widget)
        self.source_spectrogram = WaveformWidget(self, width=5, height=2)
        spec_layout.addWidget(self.source_spectrogram)
        self.source_viz_tabs.addTab(spec_widget, "Spectrogram")
        
        # MFCC tab
        mfcc_widget = QWidget()
        mfcc_layout = QVBoxLayout(mfcc_widget)
        self.source_mfcc = WaveformWidget(self, width=5, height=2)
        mfcc_layout.addWidget(self.source_mfcc)
        self.source_viz_tabs.addTab(mfcc_widget, "MFCC")
        
        viz_layout.addWidget(self.source_viz_tabs)
        
        # Output waveform
        viz_layout.addWidget(QLabel("Output Audio:"))
        self.output_viz_tabs = QTabWidget()
        
        # Waveform tab
        out_waveform_widget = QWidget()
        out_waveform_layout = QVBoxLayout(out_waveform_widget)
        self.output_waveform = WaveformWidget(self, width=5, height=2)
        out_waveform_layout.addWidget(self.output_waveform)
        self.output_viz_tabs.addTab(out_waveform_widget, "Waveform")
        
        # Spectrogram tab
        out_spec_widget = QWidget()
        out_spec_layout = QVBoxLayout(out_spec_widget)
        self.output_spectrogram = WaveformWidget(self, width=5, height=2)
        out_spec_layout.addWidget(self.output_spectrogram)
        self.output_viz_tabs.addTab(out_spec_widget, "Spectrogram")
        
        viz_layout.addWidget(self.output_viz_tabs)
        
        right_widget.addTab(viz_tab, "Visualization")
        
        # History tab
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.load_history_item)
        history_layout.addWidget(self.history_list)
        
        history_buttons = QHBoxLayout()
        self.play_history_btn = QPushButton("Play Selected")
        self.play_history_btn.clicked.connect(self.play_history_item)
        self.play_history_btn.setEnabled(False)
        
        self.save_history_btn = QPushButton("Save Selected")
        self.save_history_btn.clicked.connect(self.save_history_item)
        self.save_history_btn.setEnabled(False)
        
        self.delete_history_btn = QPushButton("Delete Selected")
        self.delete_history_btn.clicked.connect(self.delete_history_item)
        self.delete_history_btn.setEnabled(False)
        
        history_buttons.addWidget(self.play_history_btn)
        history_buttons.addWidget(self.save_history_btn)
        history_buttons.addWidget(self.delete_history_btn)
        history_layout.addLayout(history_buttons)
        
        right_widget.addTab(history_tab, "History")
        
        # Add panels to main splitter
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(center_widget)
        main_splitter.addWidget(right_widget)
        
        # Set sizes
        main_splitter.setSizes([250, 400, 350])
        
        # Set main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(main_splitter)
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Set up media player for playback
        self.media_player = QMediaPlayer()
        
        # Set the central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def record_source_audio(self):
        """Record audio from microphone as voice source"""
        try:
            self.recorder = AudioRecorder()
            self.recording_thread = RecordingThread(self.recorder)
            
            # Temporary dialog for recording
            record_dialog = QDialog(self)
            record_dialog.setWindowTitle("Record Voice Sample")
            record_dialog.setGeometry(300, 300, 400, 200)
            
            dialog_layout = QVBoxLayout()
            
            status_label = QLabel("Press Start to begin recording...")
            dialog_layout.addWidget(status_label)
            
            level_indicator = QProgressBar()
            level_indicator.setRange(0, 100)
            dialog_layout.addWidget(level_indicator)
            
            time_label = QLabel("00:00")
            dialog_layout.addWidget(time_label)
            
            button_layout = QHBoxLayout()
            start_button = QPushButton("Start Recording")
            stop_button = QPushButton("Stop Recording")
            stop_button.setEnabled(False)
            cancel_button = QPushButton("Cancel")
            
            button_layout.addWidget(start_button)
            button_layout.addWidget(stop_button)
            button_layout.addWidget(cancel_button)
            
            dialog_layout.addLayout(button_layout)
            record_dialog.setLayout(dialog_layout)
            
            # Timer for updating recording time
            start_time = None
            
            def update_timer():
                nonlocal start_time
                if start_time:
                    elapsed = datetime.now() - start_time
                    time_label.setText(f"{elapsed.seconds // 60:02}:{elapsed.seconds % 60:02}")
            
            def start_recording():
                nonlocal start_time
                start_time = datetime.now()
                status_label.setText("Recording in progress...")
                start_button.setEnabled(False)
                stop_button.setEnabled(True)
                
                # Start recording thread
                self.recording_thread.update_signal.connect(
                    lambda data: level_indicator.setValue(
                        min(100, int(max(abs(int.from_bytes(data[:2], byteorder='little', signed=True)) 
                                        for i in range(0, len(data), 2)) / 32768 * 100))
                    )
                )
                self.recording_thread.finished_signal.connect(recording_finished)
                self.recording_thread.start()
                
                # Timer updates in a simple way without another thread
                timer = QTimer()
                timer.timeout.connect(update_timer)
                timer.start(500)  # Update every 500ms
            
            def stop_recording():
                self.recording_thread.stop()
                status_label.setText("Processing recording...")
            
            def recording_finished(file_path):
                if file_path:
                    self.source_audio_path = file_path
                    self.source_label.setText(f"Recorded voice sample: {os.path.basename(file_path)}")
                    self.analyze_btn.setEnabled(True)
                    record_dialog.accept()
                else:
                    QMessageBox.warning(self, "Recording Error", 
                                      "Failed to record audio. Please check your microphone.")
                    record_dialog.reject()
            
            start_button.clicked.connect(start_recording)
            stop_button.clicked.connect(stop_recording)
            cancel_button.clicked.connect(record_dialog.reject)
            
            record_dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Recording Error", 
                               f"Error initializing recording: {str(e)}\n\nPlease make sure PyAudio is installed and your microphone is working.")

    def update_pitch_label(self, value):
        """Update the pitch slider label"""
        self.pitch_label.setText(f"Current: {value} semitones")

    def update_speed_label(self, value):
        """Update the speed slider label"""
        speed = value / 100.0
        self.speed_label.setText(f"Current: {speed:.1f}x")

    def use_preset_text(self, text):
        """Use a preset text"""
        if text != "Select a preset...":
            self.text_edit.setText(text)

    def select_source_audio(self):
        """Select an audio file for voice cloning"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Voice Sample", "", 
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)"
        )
        
        if file_path:
            self.source_audio_path = file_path
            self.source_label.setText(f"Source: {os.path.basename(file_path)}")
            self.analyze_btn.setEnabled(True)

    def analyze_audio(self):
        """Analyze the source audio to extract speaker embedding"""
        if not hasattr(self, 'source_audio_path') or not self.source_audio_path:
            QMessageBox.warning(self, "Analyze Audio", 
                              "Please select or record a voice sample first.")
            return
        
        # Disable button during analysis
        self.analyze_btn.setEnabled(False)
        self.source_progress.setValue(0)
        
        # Start analysis thread
        self.analysis_thread = AnalysisThread(self.model, self.source_audio_path)
        self.analysis_thread.progress_signal.connect(self.source_progress.setValue)
        self.analysis_thread.finished_signal.connect(self.analysis_complete)
        self.analysis_thread.start()
        
        self.statusBar().showMessage("Analyzing voice sample...")

    def analysis_complete(self, features):
        """Handle completion of voice analysis"""
        if features:
            self.speaker_embedding = features['embedding']
            self.generate_btn.setEnabled(True)
            self.save_profile_btn.setEnabled(True)
            
            # Update visualizations
            self.source_waveform.plot_waveform(features['waveform'], features['sr'])
            self.source_spectrogram.plot_spectrogram(features['waveform'], features['sr'])
            self.source_mfcc.plot_mfcc(features['mfccs'])
            
            self.statusBar().showMessage("Voice analysis complete")
        else:
            QMessageBox.warning(self, "Analysis Failed", 
                              "Failed to analyze the voice sample. Please try again with a different sample.")
            self.analyze_btn.setEnabled(True)

    def generate_speech(self):
        """Generate speech based on text and voice embedding"""
        if self.speaker_embedding is None:
            QMessageBox.warning(self, "Generate Speech", 
                              "Please analyze a voice sample first.")
            return
        
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Generate Speech", 
                              "Please enter some text to synthesize.")
            return
        
        # Get speech parameters
        pitch_shift = self.pitch_slider.value()
        speed = self.speed_slider.value() / 100.0
        emotion = self.emotion_combo.currentText()
        
        # Check batch mode
        batch_mode = self.batch_checkbox.currentText()
        
        if batch_mode == "Single Text":
            # Process as single text
            texts = [text]
        elif batch_mode == "Line by Line":
            # Split by line
            texts = [line.strip() for line in text.split('\n') if line.strip()]
        else:  # Paragraph by Paragraph
            # Split by double newline
            texts = [para.strip() for para in text.split('\n\n') if para.strip()]
        
        # Disable controls during generation
        self.generate_btn.setEnabled(False)
        self.generate_progress.setValue(0)
        
        # Process first text (or only text if single mode)
        self.current_batch_index = 0
        self.batch_texts = texts
        
        self.process_next_batch_item()
        
        self.statusBar().showMessage(f"Generating speech for {len(texts)} item(s)...")

    def process_next_batch_item(self):
        """Process the next item in batch generation"""
        if self.current_batch_index >= len(self.batch_texts):
            # Batch complete
            self.generate_btn.setEnabled(True)
            self.statusBar().showMessage("Speech generation complete")
            return
        
        # Get current text
        text = self.batch_texts[self.current_batch_index]
        
        # Get speech parameters
        pitch_shift = self.pitch_slider.value()
        speed = self.speed_slider.value() / 100.0
        emotion = self.emotion_combo.currentText()
        
        # Start synthesis thread
        self.synthesis_thread = SynthesisThread(
            self.model, self.speaker_embedding, text, pitch_shift, speed, emotion
        )
        self.synthesis_thread.progress_signal.connect(self.generate_progress.setValue)
        self.synthesis_thread.finished_signal.connect(self.synthesis_complete)
        self.synthesis_thread.start()

    def synthesis_complete(self, result):
        """Handle completion of speech synthesis"""
        y, sr = result
        
        # Store the output
        self.output_audio = y
        self.output_sr = sr
        
        # Add to history
        text = self.batch_texts[self.current_batch_index]
        history_item = VoiceHistoryItem(text, y, sr)
        self.history.append(history_item)
        
        # Update history list
        self.history_list.addItem(history_item.get_display_name())
        
        # Update visualizations
        self.output_waveform.plot_waveform(y, sr)
        self.output_spectrogram.plot_spectrogram(y, sr)
        
        # Enable buttons
        self.play_output_btn.setEnabled(True)
        self.save_output_btn.setEnabled(True)
        
        # Move to next batch item
        self.current_batch_index += 1
        
        # Update progress for overall batch
        batch_progress = int((self.current_batch_index / len(self.batch_texts)) * 100)
        self.statusBar().showMessage(f"Processing item {self.current_batch_index}/{len(self.batch_texts)} - {batch_progress}%")
        
        # Process next item if in batch mode
        if self.current_batch_index < len(self.batch_texts):
            self.process_next_batch_item()
        else:
            # All items processed
            self.generate_btn.setEnabled(True)
            self.statusBar().showMessage("Speech generation complete")

    def play_output(self):
        """Play the generated audio"""
        if self.output_audio is None:
            return
        
        # Save to temporary file for playback
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, self.output_audio, self.output_sr)
        
        # Set up media player
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(temp_file.name)))
        self.media_player.play()
        
        # Enable stop button
        self.stop_output_btn.setEnabled(True)
        self.statusBar().showMessage("Playing audio...")
        
        # Connect to media status changed
        self.media_player.mediaStatusChanged.connect(
            lambda status: self.handle_playback_status(status, temp_file.name)
        )

    def stop_output(self):
        """Stop audio playback"""
        self.media_player.stop()
        self.stop_output_btn.setEnabled(False)
        self.statusBar().showMessage("Playback stopped")

    def handle_playback_status(self, status, temp_file):
        """Handle changes in media playback status"""
        if status == QMediaPlayer.EndOfMedia:
            self.stop_output_btn.setEnabled(False)
            self.statusBar().showMessage("Playback complete")
            
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

    def save_output(self):
        """Save the generated audio to a file"""
        if self.output_audio is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio", "", "WAV Files (*.wav);;All Files (*)"
        )
        
        if file_path:
            sf.write(file_path, self.output_audio, self.output_sr)
            self.statusBar().showMessage(f"Audio saved to {file_path}")

    def load_history_item(self, item):
        """Load a history item when selected from the list"""
        idx = self.history_list.currentRow()
        if 0 <= idx < len(self.history):
            history_item = self.history[idx]
            
            # Update text
            self.text_edit.setText(history_item.text)
            
            # Update output
            self.output_audio = history_item.audio
            self.output_sr = history_item.sr
            
            # Update visualization
            self.output_waveform.plot_waveform(history_item.audio, history_item.sr)
            self.output_spectrogram.plot_spectrogram(history_item.audio, history_item.sr)
            
            # Enable buttons
            self.play_history_btn.setEnabled(True)
            self.save_history_btn.setEnabled(True)
            self.delete_history_btn.setEnabled(True)

    def play_history_item(self):
        """Play the selected history item"""
        idx = self.history_list.currentRow()
        if 0 <= idx < len(self.history):
            history_item = self.history[idx]
            
            # Save to temporary file for playback
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, history_item.audio, history_item.sr)
            
            # Set up media player
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(temp_file.name)))
            self.media_player.play()
            
            # Enable stop button
            self.stop_output_btn.setEnabled(True)
            self.statusBar().showMessage("Playing audio...")
            
            # Connect to media status changed
            self.media_player.mediaStatusChanged.connect(
                lambda status: self.handle_playback_status(status, temp_file.name)
            )

    def save_history_item(self):
        """Save the selected history item to a file"""
        idx = self.history_list.currentRow()
        if 0 <= idx < len(self.history):
            history_item = self.history[idx]
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Audio", "", "WAV Files (*.wav);;All Files (*)"
            )
            
            if file_path:
                sf.write(file_path, history_item.audio, history_item.sr)
                self.statusBar().showMessage(f"Audio saved to {file_path}")

    def delete_history_item(self):
        """Delete the selected history item"""
        idx = self.history_list.currentRow()
        if 0 <= idx < len(self.history):
            reply = QMessageBox.question(self, "Delete Item", 
                                       "Are you sure you want to delete this item?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.history.pop(idx)
                self.history_list.takeItem(idx)
                
                # Disable buttons if list is empty
                if not self.history:
                    self.play_history_btn.setEnabled(False)
                    self.save_history_btn.setEnabled(False)
                    self.delete_history_btn.setEnabled(False)
                    
    def new_profile(self):
        """Create a new voice profile"""
        name, ok = QInputDialog.getText(
            self, "New Voice Profile", "Enter name for new profile:"
        )
        
        if ok and name:
            if name in self.voice_profiles:
                QMessageBox.warning(self, "New Profile", 
                                  f"A profile named '{name}' already exists.")
                return
                
            self.voice_profiles[name] = None
            self.profile_combo.addItem(name)
            self.profile_combo.setCurrentText(name)
            self.current_voice_profile = name
            
            self.statusBar().showMessage(f"Created new profile: {name}")

    def save_profile(self):
        """Save current embedding to the active profile"""
        if self.speaker_embedding is None:
            QMessageBox.warning(self, "Save Profile", 
                              "Please analyze a voice sample first.")
            return
            
        # Save the embedding to current profile
        self.voice_profiles[self.current_voice_profile] = self.speaker_embedding.clone()
        
        self.statusBar().showMessage(f"Voice saved to profile: {self.current_voice_profile}")

    def change_profile(self, profile_name):
        """Switch to a different voice profile"""
        if profile_name in self.voice_profiles:
            self.current_voice_profile = profile_name
            embedding = self.voice_profiles[profile_name]
            
            if embedding is not None:
                self.speaker_embedding = embedding.clone()
                self.generate_btn.setEnabled(True)
                self.save_profile_btn.setEnabled(True)
                self.statusBar().showMessage(f"Switched to profile: {profile_name}")
            else:
                self.save_profile_btn.setEnabled(True)
                self.statusBar().showMessage(f"Selected empty profile: {profile_name}")

# For QTimer
from PyQt5.QtCore import QTimer

# Main entry point
if __name__ == '__main__':
    print("Starting Voice Cloning App...")
    app = QApplication(sys.argv)
    window = ProductionVoiceCloningApp()
    window.show()
    print("Application window should be visible now")
    sys.exit(app.exec_())
