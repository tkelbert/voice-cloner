#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# voice_cloning_app.py - Complete voice cloning application with multiple TTS models

# Imports
import io
from pydub import AudioSegment
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
import subprocess
import logging
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                           QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, 
                           QSlider, QProgressBar, QMessageBox, QGroupBox,
                           QTextEdit, QTabWidget, QSplitter, QComboBox,
                           QListWidget, QDialog, QInputDialog, QLineEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("voice_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VoiceApp")

try:
    import pyaudio
    logger.info("PyAudio imported successfully")
except ImportError:
    logger.warning("PyAudio not found. Recording functionality will be disabled.")
    # Create a dummy pyaudio to prevent crashes
    class DummyPyAudio:
        def __init__(self):
            pass
        def open(self, *args, **kwargs):
            raise Exception("PyAudio not installed")
    pyaudio = type('DummyModule', (), {'PyAudio': DummyPyAudio, 'paInt16': 16})

# Base voice model classes
class BaseVoiceModel(ABC):
    """Abstract base class for all voice models"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the model, returns True if successful"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if model is available/downloaded"""
        pass
    
    @abstractmethod
    def needs_download(self) -> bool:
        """Check if model needs to be downloaded"""
        pass
        
    @abstractmethod
    def download_model(self, progress_callback=None) -> bool:
        """Download the model if needed, with optional progress callback"""
        pass
        
    @abstractmethod
    def extract_speaker_embedding(self, audio_path: str) -> Dict[str, Any]:
        """Extract speaker characteristics from audio file"""
        pass
        
    @abstractmethod
    def synthesize_speech(self, embedding: Any, text: str, 
                          pitch_shift: float = 0, speed: float = 1.0, 
                          emotion: str = "neutral") -> Tuple[np.ndarray, int]:
        """Synthesize speech using the model"""
        pass
    
    def _create_dummy_features(self, sample_rate: int = 22050) -> Dict[str, Any]:
        """Create dummy features when analysis fails"""
        logger.warning("Creating dummy features due to analysis failure")
        # Create a simple sine wave as dummy waveform
        duration = 1.0  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        dummy_waveform = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Create dummy MFCCs
        dummy_mfccs = np.random.randn(20, 100)  # 20 MFCC coefficients, 100 time frames
        
        return {
            'embedding': None,  # Specific to each model implementation
            'waveform': dummy_waveform,
            'sr': sample_rate,
            'mfccs': dummy_mfccs
        }

class PlaceholderModel(BaseVoiceModel):
    """The original placeholder model that produces sonar-like sounds"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.sample_rate = 22050
        self.initialized = True
        logger.info(f"Placeholder voice model initialized on {device}")
        
        # Placeholder model parameters - fixed shapes to match MFCC dimensions
        # MFCCs typically have shape (n_mfcc, time_frames)
        # We're using 20 MFCCs, so the encoder input will be 20
        self.encoder_params = torch.randn(20, 32).to(device)
        self.decoder_params = torch.randn(32, 16).to(device)
        self.vocoder_params = torch.randn(16, 8).to(device)
    
    def initialize(self) -> bool:
        # Already initialized in __init__
        return self.initialized
    
    def is_available(self) -> bool:
        # Placeholder is always available
        return True
    
    def needs_download(self) -> bool:
        # Placeholder doesn't need downloading
        return False
    
    def download_model(self, progress_callback=None) -> bool:
        # Nothing to download
        if progress_callback:
            progress_callback(100)
        return True
    
    def extract_speaker_embedding(self, audio_path: str) -> Dict[str, Any]:
        """Extract speaker embedding from audio file"""
        try:
            logger.info(f"Extracting speaker embedding from {audio_path}")
            
            # Check if it's an MP3 file
            is_mp3 = audio_path.lower().endswith('.mp3')
            
            try:
                # First try with librosa
                y, sr = librosa.load(audio_path, sr=self.sample_rate)
                logger.debug(f"Loaded audio with librosa: length={len(y)}, sample_rate={sr}")
            except Exception as e:
                if is_mp3:
                    logger.warning(f"Librosa failed to load MP3, trying pydub fallback: {str(e)}")
                    try:
                        # Load with pydub
                        audio = AudioSegment.from_mp3(audio_path)
                        # Convert to numpy array
                        y = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
                        
                        # If stereo, convert to mono
                        if audio.channels == 2:
                            y = y.reshape(-1, 2).mean(axis=1)
                        
                        # Get sample rate
                        sr = audio.frame_rate
                        
                        # Resample if needed
                        if sr != self.sample_rate:
                            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
                            sr = self.sample_rate
                        
                        logger.debug(f"Loaded MP3 with pydub fallback: length={len(y)}, sample_rate={sr}")
                    except Exception as pydub_error:
                        logger.error(f"Pydub fallback also failed: {str(pydub_error)}")
                        return self._create_dummy_features()
                else:
                    logger.error(f"Error loading audio: {str(e)}")
                    return self._create_dummy_features()
            
            # Extract features (MFCC for demonstration)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # Normalize
            mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
            
            # Convert to tensor
            mfccs_tensor = torch.FloatTensor(mfccs).to(self.device)
            
            # Simulate encoder output - properly handle dimension
            embedding = torch.matmul(torch.mean(mfccs_tensor, dim=1).unsqueeze(0), self.encoder_params)
            logger.debug(f"Created embedding with shape {embedding.shape}")
            
            # Also return the audio for visualization
            return {
                'embedding': embedding,
                'waveform': y,
                'sr': sr,
                'mfccs': mfccs
            }
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {e}", exc_info=True)
            return self._create_dummy_features()
    
    def synthesize_speech(self, embedding, text, pitch_shift=0, speed=1.0, emotion="neutral"):
        """Synthesize speech based on speaker embedding and text"""
        try:
            logger.info(f"Synthesizing speech with placeholder model: '{text[:30]}...'")
            
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
            
            logger.debug(f"Generated audio: length={len(y)}, min={y.min():.2f}, max={y.max():.2f}")
            return y, self.sample_rate
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}", exc_info=True)
            # Return a simple beep tone as a fallback
            t = np.linspace(0, 1.0, self.sample_rate, endpoint=False)
            return np.sin(2 * np.pi * 440 * t), self.sample_rate

class MozillaTTSModel(BaseVoiceModel):
    """Mozilla TTS voice model implementation"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.sample_rate = 22050
        self.tts = None
        self.encoder = None
        self.vocoder = None
        self.initialized = False
        self.model_path = os.path.join(os.path.expanduser("~"), ".tts_models")
        logger.info(f"Mozilla TTS model object created on {device}")
    
    def initialize(self) -> bool:
        """Initialize the Mozilla TTS model"""
        try:
            if self.initialized:
                return True
            
            logger.info("Initializing Mozilla TTS model")
            import TTS
            from TTS.api import TTS as TTS_API
            
            # Create model directory if it doesn't exist
            os.makedirs(self.model_path, exist_ok=True)
            
            # Initialize TTS with voice cloning capability
            self.tts = TTS_API(model_name="tts_models/multilingual/multi-dataset/your_tts", 
                          progress_bar=False,
                          gpu=torch.cuda.is_available())
            
            # Encoder for speaker embeddings
            logger.info("Mozilla TTS model initialized successfully")
            self.initialized = True
            return True
        except ImportError:
            logger.error("TTS package not installed. Please install with 'pip install TTS'")
            return False
        except Exception as e:
            logger.error(f"Error initializing Mozilla TTS: {str(e)}", exc_info=True)
            return False
    
    def is_available(self) -> bool:
        """Check if model is available"""
        try:
            import TTS
            return True
        except ImportError:
            return False
    
    def needs_download(self) -> bool:
        """Check if model needs to be downloaded"""
        if not self.is_available():
            return True
            
        # Check if model files exist
        try:
            import TTS
            from TTS.api import TTS as TTS_API
            
            # Just try to initialize - TTS will download if needed
            # We don't actually initialize here, we just check if it works
            test_model = TTS_API(model_name="tts_models/multilingual/multi-dataset/your_tts", 
                           progress_bar=False,
                           gpu=False,
                           load_config_only=True)
            return False
        except:
            return True
    
    def download_model(self, progress_callback=None) -> bool:
        """Download the Mozilla TTS model"""
        try:
            if not self.is_available():
                logger.error("TTS package not installed. Cannot download model.")
                return False
                
            def download_thread():
                try:
                    # Import inside thread to avoid issues
                    import TTS
                    from TTS.api import TTS as TTS_API
                    
                    # This will download model if not already downloaded
                    self.tts = TTS_API(model_name="tts_models/multilingual/multi-dataset/your_tts", 
                                  progress_bar=True,
                                  gpu=torch.cuda.is_available())
                    
                    if progress_callback:
                        progress_callback(100)
                        
                    self.initialized = True
                    logger.info("Mozilla TTS model downloaded successfully")
                except Exception as e:
                    logger.error(f"Error downloading Mozilla TTS model: {str(e)}", exc_info=True)
            
            # Start download in separate thread
            thread = threading.Thread(target=download_thread)
            thread.daemon = True
            thread.start()
            
            if progress_callback:
                progress_callback(10)  # Initial progress
                
            return True
        except Exception as e:
            logger.error(f"Error in download process: {str(e)}", exc_info=True)
            return False
    
    def extract_speaker_embedding(self, audio_path: str) -> Dict[str, Any]:
        """Extract speaker embedding from audio file"""
        if not self.initialize():
            logger.error("Failed to initialize Mozilla TTS model")
            return self._create_dummy_features()
            
        try:
            logger.info(f"Extracting speaker embedding with Mozilla TTS from {audio_path}")
            
            # Load audio for visualization
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract MFCC for visualization
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # For Mozilla TTS/YourTTS, we don't need to extract embedding separately
            # We just need the audio path for the synthesize_speech function
            # But we do set up an embedding object to store the audio path
            embedding = {
                "audio_path": audio_path
            }
            
            return {
                'embedding': embedding,
                'waveform': y,
                'sr': sr,
                'mfccs': mfccs
            }
        except Exception as e:
            logger.error(f"Error extracting speaker embedding with Mozilla TTS: {e}", exc_info=True)
            return self._create_dummy_features()
    
    def synthesize_speech(self, embedding, text, pitch_shift=0, speed=1.0, emotion="neutral"):
        """Synthesize speech using Mozilla TTS"""
        if not self.initialize():
            logger.error("Failed to initialize Mozilla TTS model")
            t = np.linspace(0, 1.0, self.sample_rate, endpoint=False)
            return np.sin(2 * np.pi * 440 * t), self.sample_rate
            
        try:
            logger.info(f"Synthesizing speech with Mozilla TTS: '{text[:30]}...'")
            
            # Get audio path from embedding
            audio_path = embedding["audio_path"]
            
            # Generate speech
            wav = self.tts.tts(
                text=text,
                speaker_wav=audio_path,
                language="en"
            )
            
            # Convert to numpy array if it's not already
            if not isinstance(wav, np.ndarray):
                wav = np.array(wav)
            
            # Apply pitch shift if requested
            if pitch_shift != 0:
                wav = librosa.effects.pitch_shift(wav, sr=self.tts.synthesizer.output_sample_rate, n_steps=pitch_shift)
            
            # Apply speed change if requested
            if speed != 1.0:
                wav = librosa.effects.time_stretch(wav, rate=speed)
            
            logger.debug(f"Generated audio with Mozilla TTS: length={len(wav)}")
            return wav, self.tts.synthesizer.output_sample_rate
            
        except Exception as e:
            logger.error(f"Error synthesizing speech with Mozilla TTS: {e}", exc_info=True)
            # Return a simple beep tone as a fallback
            t = np.linspace(0, 1.0, self.sample_rate, endpoint=False)
            return np.sin(2 * np.pi * 440 * t), self.sample_rate

class CoquiXTTSModel(BaseVoiceModel):
    """Coqui XTTS voice model implementation"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.sample_rate = 24000  # XTTS default
        self.model = None
        self.initialized = False
        self.model_path = os.path.join(os.path.expanduser("~"), ".tts_models")
        logger.info(f"Coqui XTTS model object created on {device}")
    
    def initialize(self) -> bool:
        """Initialize the Coqui XTTS model"""
        try:
            if self.initialized:
                return True
            
            logger.info("Initializing Coqui XTTS model")
            
            try:
                # Try importing with new Coqui TTS structure
                from TTS.api import TTS
                self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            except:
                # Fall back to manual loading for older versions
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import Xtts
                
                # Create model directory if it doesn't exist
                os.makedirs(self.model_path, exist_ok=True)
                
                # Set up config and model
                config = XttsConfig()
                self.model = Xtts.init_from_config(config)
                model_path = os.path.join(self.model_path, "xtts_v2")
                if os.path.exists(model_path):
                    self.model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
                    if torch.cuda.is_available():
                        self.model = self.model.cuda()
                else:
                    logger.error(f"XTTS model not found at {model_path}. Please download it first.")
                    return False
            
            logger.info("Coqui XTTS model initialized successfully")
            self.initialized = True
            return True
        except ImportError:
            logger.error("TTS package not installed. Please install with 'pip install TTS'")
            return False
        except Exception as e:
            logger.error(f"Error initializing Coqui XTTS: {str(e)}", exc_info=True)
            return False
    
    def is_available(self) -> bool:
        """Check if model is available"""
        try:
            import TTS
            return True
        except ImportError:
            return False
    
    def needs_download(self) -> bool:
        """Check if model needs to be downloaded"""
        if not self.is_available():
            return True
            
        # Check if model directory exists
        try:
            # Try with new API structure first
            from TTS.api import TTS
            try:
                # This will check if model exists
                TTS("tts_models/multilingual/multi-dataset/xtts_v2", verbose=False)
                return False
            except:
                return True
        except:
            # Older model structure
            model_path = os.path.join(self.model_path, "xtts_v2")
            return not os.path.exists(model_path)
    
    def download_model(self, progress_callback=None) -> bool:
        """Download the Coqui XTTS model"""
        try:
            if not self.is_available():
                logger.error("TTS package not installed. Cannot download model.")
                return False
                
            def download_thread():
                try:
                    # Import inside thread to avoid issues
                    from TTS.api import TTS
                    
                    # This will download model if not already present
                    self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                    
                    if progress_callback:
                        progress_callback(100)
                        
                    self.initialized = True
                    logger.info("Coqui XTTS model downloaded successfully")
                except Exception as e:
                    logger.error(f"Error downloading Coqui XTTS model: {str(e)}", exc_info=True)
            
            # Start download in separate thread
            thread = threading.Thread(target=download_thread)
            thread.daemon = True
            thread.start()
            
            if progress_callback:
                progress_callback(10)  # Initial progress
                
            return True
        except Exception as e:
            logger.error(f"Error in download process: {str(e)}", exc_info=True)
            return False
    
    def extract_speaker_embedding(self, audio_path: str) -> Dict[str, Any]:
        """Extract speaker embedding for Coqui XTTS"""
        if not self.initialize():
            logger.error("Failed to initialize Coqui XTTS model")
            return self._create_dummy_features()
            
        try:
            logger.info(f"Extracting speaker embedding with Coqui XTTS from {audio_path}")
            
            # Load audio for visualization
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract MFCC for visualization
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # For XTTS, we don't need to extract embedding separately
            # We just need the audio path for the synthesize_speech function
            # But we do set up an embedding object to store the audio path
            embedding = {
                "audio_path": audio_path,
                "speaker_wav": y,
                "sample_rate": sr
            }
            
            return {
                'embedding': embedding,
                'waveform': y,
                'sr': sr,
                'mfccs': mfccs
            }
        except Exception as e:
            logger.error(f"Error extracting speaker embedding with Coqui XTTS: {e}", exc_info=True)
            return self._create_dummy_features()
    
    def synthesize_speech(self, embedding, text, pitch_shift=0, speed=1.0, emotion="neutral"):
        """Synthesize speech using Coqui XTTS"""
        if not self.initialize():
            logger.error("Failed to initialize Coqui XTTS model")
            t = np.linspace(0, 1.0, self.sample_rate, endpoint=False)
            return np.sin(2 * np.pi * 440 * t), self.sample_rate
            
        try:
            logger.info(f"Synthesizing speech with Coqui XTTS: '{text[:30]}...'")
            
            # Get audio path from embedding
            audio_path = embedding["audio_path"]
            
            # Use different approach depending on API version
            try:
                # Try with TTS class (newer API)
                wav = self.model.tts(
                    text=text,
                    speaker_wav=audio_path,
                    language="en"
                )
            except AttributeError:
                # Fall back to direct model call for older API
                if "speaker_wav" in embedding and "sample_rate" in embedding:
                    # Use pre-loaded audio if available
                    gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                        embedding["speaker_wav"], 
                        embedding["sample_rate"]
                    )
                else:
                    # Load from file if needed
                    speaker_wav, speaker_sr = librosa.load(audio_path, sr=self.sample_rate)
                    gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                        speaker_wav, 
                        speaker_sr
                    )
                    
                # Apply emotion through text instructions if requested
                if emotion != "neutral":
                    emotion_prompts = {
                        "happy": " [speaking with joy and happiness]",
                        "sad": " [speaking with sadness]",
                        "angry": " [speaking with anger]",
                        "surprised": " [speaking with surprise]"
                    }
                    text = text + emotion_prompts.get(emotion, "")
                
                # Generate with XTTS
                wav = self.model.inference(
                    text,
                    "en",
                    gpt_cond_latent,
                    speaker_embedding,
                    temperature=0.7,
                )
            
            # Convert to numpy array if it's not already
            if not isinstance(wav, np.ndarray):
                wav = np.array(wav)
            
            # Apply pitch shift if requested
            if pitch_shift != 0:
                wav = librosa.effects.pitch_shift(wav, sr=self.sample_rate, n_steps=pitch_shift)
            
            # Apply speed change if requested
            if speed != 1.0:
                wav = librosa.effects.time_stretch(wav, rate=speed)
            
            logger.debug(f"Generated audio with Coqui XTTS: length={len(wav)}")
            return wav, self.sample_rate
            
        except Exception as e:
            logger.error(f"Error synthesizing speech with Coqui XTTS: {e}", exc_info=True)
            # Return a simple beep tone as a fallback
            t = np.linspace(0, 1.0, self.sample_rate, endpoint=False)
            return np.sin(2 * np.pi * 440 * t), self.sample_rate

class VoiceModelFactory:
    """Factory class to create appropriate voice models"""
    
    @staticmethod
    def create_model(model_type, device='cuda' if torch.cuda.is_available() else 'cpu'):
        if model_type == 0 or model_type == "Placeholder":
            return PlaceholderModel(device)
        elif model_type == 1 or model_type == "Mozilla TTS":
            return MozillaTTSModel(device)
        elif model_type == 2 or model_type == "Coqui XTTS":
            return CoquiXTTSModel(device)
        else:
            logger.warning(f"Unknown model type: {model_type}, using placeholder model")
            return PlaceholderModel(device)

# Simple audio player class that doesn't rely on QtMultimedia
class SimpleAudioPlayer:
    def __init__(self):
        self.current_process = None
        self.current_temp_file = None
        logger.info("Audio player initialized")
        
    def play(self, file_path):
        logger.debug(f"Attempting to play audio file: {file_path}")
        self.stop()  # Stop any existing playback
        
        # Try to find a suitable audio player command
        players = [
            ['aplay', file_path],  # Linux
            ['ffplay', '-nodisp', '-autoexit', file_path],  # FFmpeg
            ['mplayer', file_path],  # MPlayer
            ['mpg123', file_path],  # MPG123
            ['play', file_path],    # SoX
            ['paplay', file_path],  # PulseAudio
        ]
        
        for player_cmd in players:
            try:
                # Check if the command exists
                which_process = subprocess.run(['which', player_cmd[0]], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
                
                if which_process.returncode == 0:
                    logger.debug(f"Found player: {player_cmd[0]}")
                    # Start the player in background
                    self.current_process = subprocess.Popen(
                        player_cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE
                    )
                    logger.info(f"Started audio playback with {player_cmd[0]}")
                    return True
            except Exception as e:
                logger.error(f"Error trying to use {player_cmd[0]}: {str(e)}")
                continue
                
        logger.error("No suitable audio player found")
        print("No suitable audio player found. Please install one of: aplay, ffplay, mplayer, mpg123, play, or paplay")
        return False
    
    def stop(self):
        if self.current_process and self.current_process.poll() is None:
            logger.debug("Stopping current audio playback")
            self.current_process.terminate()
            self.current_process = None
        
        # Clean up temp file if it exists
        if self.current_temp_file:
            try:
                os.unlink(self.current_temp_file)
                logger.debug(f"Deleted temporary file: {self.current_temp_file}")
            except Exception as e:
                logger.error(f"Error deleting temp file: {str(e)}")
            self.current_temp_file = None

# Voice synthesis model with multiple implementation support
class VoiceSynthesisModel:
    """Voice synthesis model that wraps different implementations"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.sample_rate = 22050
        logger.info(f"Voice model initialized on {device}")
        
        # Initialize with placeholder model
        self.current_model_type = 0  # Placeholder
        self.model = VoiceModelFactory.create_model(self.current_model_type, device)
    
    def change_model(self, model_type):
        """Change the current model type"""
        if self.current_model_type == model_type:
            return True  # Already using this model
            
        logger.info(f"Changing voice model from {self.current_model_type} to {model_type}")
        self.current_model_type = model_type
        self.model = VoiceModelFactory.create_model(model_type, self.device)
        return self.model.is_available()
    
    def get_model_status(self):
        """Get the status of the current model"""
        if not self.model.is_available():
            return "Not installed"
        elif self.model.needs_download():
            return "Needs download"
        elif not getattr(self.model, "initialized", False):
            return "Not initialized"
        else:
            return "Ready"
    
    def download_model(self, progress_callback=None):
        """Download the current model if needed"""
        return self.model.download_model(progress_callback)
    
    def extract_speaker_embedding(self, audio_path):
        """Extract speaker embedding from audio file using the current model"""
        return self.model.extract_speaker_embedding(audio_path)
    
    def synthesize_speech(self, embedding, text, pitch_shift=0, speed=1.0, emotion="neutral"):
        """Synthesize speech using the current model"""
        return self.model.synthesize_speech(embedding, text, pitch_shift, speed, emotion)

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
        logger.info("AudioRecorder initialized")
    
    def start_recording(self):
        try:
            logger.info("Starting audio recording")
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
            
            logger.debug("Recording started")
        except Exception as e:
            logger.error(f"Error starting recording: {e}", exc_info=True)
            self.is_recording = False
            raise
    
    def stop_recording(self):
        logger.info("Stopping recording")
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.p:
            self.p.terminate()
            
        self.is_recording = False
        
        # Create a temporary file to store the recording
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
        wf = wave.open(self.temp_file.name, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        logger.debug(f"Recording saved to {self.temp_file.name}")
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
        logger.debug("RecordingThread initialized")
    
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
                    logger.error(f"Error during recording: {e}")
                    break
                    
            file_path = self.recorder.stop_recording()
            self.finished_signal.emit(file_path)
            logger.info("Recording thread finished")
        except Exception as e:
            logger.error(f"Recording thread error: {e}", exc_info=True)
            self.finished_signal.emit("")
    
    def stop(self):
        logger.debug("Stopping recording thread")
        self.running = False

class AnalysisThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)
    
    def __init__(self, model, audio_path):
        super().__init__()
        self.model = model
        self.audio_path = audio_path
        logger.debug(f"AnalysisThread initialized for {audio_path}")
        
    def run(self):
        logger.info("Starting analysis thread")
        # Progress updates
        for i in range(101):
            self.progress_signal.emit(i)
            self.msleep(10)
        
        # Perform analysis
        features = self.model.extract_speaker_embedding(self.audio_path)
        self.finished_signal.emit(features)
        logger.debug("Analysis thread finished")

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
        logger.debug(f"SynthesisThread initialized for text: {text[:30]}...")
        
    def run(self):
        logger.info("Starting synthesis thread")
        # Progress updates
        for i in range(101):
            self.progress_signal.emit(i)
            self.msleep(20)
        
        # Perform synthesis
        y, sr = self.model.synthesize_speech(
            self.embedding, self.text, self.pitch_shift, self.speed, self.emotion)
        self.finished_signal.emit((y, sr))
        logger.debug("Synthesis thread finished")

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
        logger.debug("WaveformWidget initialized")
        
    def plot_waveform(self, y, sr):
        logger.debug(f"Plotting waveform: length={len(y)}, sr={sr}")
        self.ax.clear()
        time = np.arange(0, len(y)) / sr
        self.ax.plot(time, y, color='#00A8E8')
        self.ax.set_title('Waveform', color='white')
        self.ax.set_xlabel('Time (s)', color='white')
        self.ax.set_ylabel('Amplitude', color='white')
        self.fig.tight_layout()
        self.draw()
        
    def plot_spectrogram(self, y, sr):
        logger.debug(f"Plotting spectrogram: length={len(y)}, sr={sr}")
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
        logger.debug(f"Plotting MFCC: shape={mfccs.shape}")
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
        logger.debug(f"Created history item: {self.get_display_name()}")
    
    def get_display_name(self):
        return f"{self.timestamp.strftime('%H:%M:%S')} - {self.text[:30]}{'...' if len(self.text) > 30 else ''}"

class ProductionVoiceCloningApp(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Initializing Voice Cloning App")
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
        
        # Initialize audio player
        self.audio_player = SimpleAudioPlayer()
        
        self.init_ui()
        logger.info("UI initialization complete")
        
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
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Voice Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Placeholder", "Mozilla TTS", "Coqui XTTS"])
        self.model_combo.setCurrentIndex(0)  # Default to placeholder
        self.model_combo.currentIndexChanged.connect(self.change_voice_model)
        model_layout.addWidget(self.model_combo)
        synthesis_layout.addLayout(model_layout)
        
        # Model download status
        self.model_status_label = QLabel("Model status: Ready")
        synthesis_layout.addWidget(self.model_status_label)
        
        # Download model button
        self.download_model_btn = QPushButton("Download Selected Model")
        self.download_model_btn.clicked.connect(self.download_voice_model)
        self.download_model_btn.setEnabled(False)
        synthesis_layout.addWidget(self.download_model_btn)
        
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
        output_group.setObjectName("Output Controls")  # Set object name for finding later
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
        
        # Test audio button
        self.test_audio_btn = QPushButton("Test Audio System")
        self.test_audio_btn.clicked.connect(self.test_audio_system)
        output_layout.addWidget(self.test_audio_btn)
        
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
        
        # Set the central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def change_voice_model(self, model_index):
        """Change the voice synthesis model"""
        if model_index < 0 or model_index > 2:
            logger.warning(f"Invalid model index: {model_index}")
            return
            
        logger.info(f"Changing to voice model: {model_index}")
        
        # Update model in voice synthesis model
        model_available = self.model.change_model(model_index)
        
        # Update model status
        status = self.model.get_model_status()
        self.model_status_label.setText(f"Model status: {status}")
        
        # Enable/disable download button
        self.download_model_btn.setEnabled(status == "Needs download")
        
        # Show warning if model not available
        if not model_available:
            QMessageBox.warning(self, "Model Not Available", 
                              "This model is not installed. Please install the required package.")
    
    def download_voice_model(self):
        """Download the selected voice model"""
        logger.info("Starting model download")
        
        # Create progress dialog
        progress_dialog = QDialog(self)
        progress_dialog.setWindowTitle("Downloading Model")
        progress_dialog.setFixedSize(300, 100)
        progress_layout = QVBoxLayout()
        
        status_label = QLabel("Downloading model, please wait...")
        progress_layout.addWidget(status_label)
        
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_layout.addWidget(progress_bar)
        
        progress_dialog.setLayout(progress_layout)
        
        # Callback for progress updates
        def update_progress(value):
            progress_bar.setValue(value)
            if value >= 100:
                status_label.setText("Download complete!")
                self.model_status_label.setText(f"Model status: {self.model.get_model_status()}")
                # Close dialog after a delay
                QTimer.singleShot(1000, progress_dialog.accept)
        
        # Start download
        self.model.download_model(update_progress)
        
        # Show dialog
        progress_dialog.exec_()
    
    def test_audio_system(self):
        """Generate and play a test tone to verify audio playback"""
        logger.info("Testing audio system")
        import numpy as np
        import soundfile as sf
        import tempfile
        
        # Create a simple beep
        sample_rate = 22050
        duration = 1  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        beep = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Save as temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, beep, sample_rate)
        logger.debug(f"Test tone saved to {temp_file.name}")
        
        # Try to play it
        if self.audio_player.play(temp_file.name):
            self.statusBar().showMessage("Playing test tone...")
            self.audio_player.current_temp_file = temp_file.name
            logger.info("Test tone playing successfully")
            print("Test tone playing. You should hear a 1-second beep.")
        else:
            logger.error("Failed to play test tone - no audio player found")
            QMessageBox.warning(self, "Audio System Error", 
                              "No audio player found. Install aplay, ffplay, or mplayer.")
            print("Failed to play test tone. No suitable audio player found.")
            
    def record_source_audio(self):
        """Record audio from microphone as voice source"""
        try:
            logger.info("Starting recording UI")
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
            timer = QTimer()
            
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
                
                # Timer updates
                timer.timeout.connect(update_timer)
                timer.start(500)  # Update every 500ms
                logger.debug("Recording started")
            
            def stop_recording():
                logger.debug("Stopping recording")
                self.recording_thread.stop()
                status_label.setText("Processing recording...")
                timer.stop()
            
            def recording_finished(file_path):
                if file_path:
                    logger.info(f"Recording saved to {file_path}")
                    self.source_audio_path = file_path
                    self.source_label.setText(f"Recorded voice sample: {os.path.basename(file_path)}")
                    self.analyze_btn.setEnabled(True)
                    record_dialog.accept()
                else:
                    logger.error("Recording failed")
                    QMessageBox.warning(self, "Recording Error", 
                                      "Failed to record audio. Please check your microphone.")
                    record_dialog.reject()
            
            start_button.clicked.connect(start_recording)
            stop_button.clicked.connect(stop_recording)
            cancel_button.clicked.connect(record_dialog.reject)
            
            record_dialog.exec_()
        except Exception as e:
            logger.error(f"Error initializing recording: {str(e)}", exc_info=True)
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
            logger.debug(f"Using preset text: {text}")
            self.text_edit.setText(text)

    def select_source_audio(self):
        """Select an audio file for voice cloning"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Voice Sample", "", 
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)"
        )
        
        if file_path:
            logger.info(f"Selected source audio: {file_path}")
            self.source_audio_path = file_path
            self.source_label.setText(f"Source: {os.path.basename(file_path)}")
            self.analyze_btn.setEnabled(True)

    def analyze_audio(self):
        """Analyze the source audio to extract speaker embedding"""
        if not hasattr(self, 'source_audio_path') or not self.source_audio_path:
            logger.warning("Attempted analysis without source audio")
            QMessageBox.warning(self, "Analyze Audio", 
                              "Please select or record a voice sample first.")
            return
        
        logger.info(f"Starting analysis of {self.source_audio_path}")
        
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
            logger.info("Analysis completed successfully")
            self.speaker_embedding = features['embedding']
            
            # Log embedding info
            logger.debug(f"Embedding shape: {features['embedding'].shape if hasattr(features['embedding'], 'shape') else 'Not a tensor'}")
            logger.debug(f"Embedding type: {type(features['embedding'])}")
            
            # Enable generate button
            self.generate_btn.setEnabled(True)
            logger.debug("Generate button enabled")
            
            self.save_profile_btn.setEnabled(True)
            
            # Update visualizations
            self.source_waveform.plot_waveform(features['waveform'], features['sr'])
            self.source_spectrogram.plot_spectrogram(features['waveform'], features['sr'])
            self.source_mfcc.plot_mfcc(features['mfccs'])
            
            self.statusBar().showMessage("Voice analysis complete")
            logger.info("UI updated after analysis")
        else:
            logger.error("Analysis failed - features is None")
            QMessageBox.warning(self, "Analysis Failed", 
                              "Failed to analyze the voice sample. Please try again with a different sample.")
            self.analyze_btn.setEnabled(True)

    def generate_speech(self):
        """Generate speech based on text and voice embedding"""
        if self.speaker_embedding is None:
            logger.warning("Attempted to generate speech without embedding")
            QMessageBox.warning(self, "Generate Speech", 
                              "Please analyze a voice sample first.")
            return
        
        text = self.text_edit.toPlainText().strip()
        if not text:
            logger.warning("Attempted to generate speech with empty text")
            QMessageBox.warning(self, "Generate Speech", 
                              "Please enter some text to synthesize.")
            return
        
        logger.info(f"Starting speech generation with text: {text[:30]}...")
        
        # Get speech parameters
        pitch_shift = self.pitch_slider.value()
        speed = self.speed_slider.value() / 100.0
        emotion = self.emotion_combo.currentText()
        logger.debug(f"Parameters: pitch={pitch_shift}, speed={speed}, emotion={emotion}")
        
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
        
        logger.debug(f"Batch mode: {batch_mode}, number of texts: {len(texts)}")
        
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
            logger.info("All batch items processed")
            return
        
        # Get current text
        text = self.batch_texts[self.current_batch_index]
        logger.debug(f"Processing batch item {self.current_batch_index+1}/{len(self.batch_texts)}: {text[:30]}...")
        
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
        
        # Debug info
        logger.info(f"Synthesis complete: audio length={len(y)}, sample rate={sr}")
        logger.debug(f"Audio min/max values: {y.min():.4f}/{y.max():.4f}")
        
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
        logger.debug("Play button enabled")
        
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
            logger.info("Speech generation complete for all items")

    def play_output(self):
        """Play the generated audio"""
        if self.output_audio is None:
            logger.warning("Attempted to play output when no output audio available")
            return
        
        # Save to temporary file for playback
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        logger.debug(f"Saving temporary audio file to {temp_file.name}")
        
        try:
            sf.write(temp_file.name, self.output_audio, self.output_sr)
            logger.debug(f"Audio saved to temporary file: {temp_file.name}")
            
            # Play using simple audio player
            if self.audio_player.play(temp_file.name):
                # Enable stop button and update status
                self.stop_output_btn.setEnabled(True)
                self.statusBar().showMessage("Playing audio...")
                self.audio_player.current_temp_file = temp_file.name
                logger.info("Audio playback started")
            else:
                logger.error("No suitable audio player found")
                QMessageBox.warning(self, "Playback Error", 
                                  "No suitable audio player found. Please install 'aplay', 'ffplay', or 'mplayer'.")
        except Exception as e:
            logger.error(f"Error during audio playback: {str(e)}", exc_info=True)
            QMessageBox.warning(self, "Playback Error", f"Error playing audio: {str(e)}")

    def stop_output(self):
        """Stop audio playback"""
        logger.debug("Stopping audio playback")
        self.audio_player.stop()
        self.stop_output_btn.setEnabled(False)
        self.statusBar().showMessage("Playback stopped")

    def save_output(self):
        """Save the generated audio to a file"""
        if self.output_audio is None:
            logger.warning("Attempted to save output when no audio available")
            return
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Audio", "", "WAV Files (*.wav);;MP3 Files (*.mp3);;All Files (*)"
        )
        
        if file_path:
            try:
                # Determine format based on extension or selected filter
                is_mp3 = file_path.lower().endswith('.mp3') or 'MP3' in selected_filter
                
                if is_mp3:
                    logger.info(f"Saving audio as MP3 to {file_path}")
                    # First save as temporary WAV
                    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    sf.write(temp_wav.name, self.output_audio, self.output_sr)
                    
                    # Convert to MP3 using pydub
                    audio = AudioSegment.from_wav(temp_wav.name)
                    audio.export(file_path, format="mp3")
                    
                    # Clean up temp file
                    os.unlink(temp_wav.name)
                else:
                    # Save as WAV
                    logger.info(f"Saving audio to {file_path}")
                    sf.write(file_path, self.output_audio, self.output_sr)
                    
                self.statusBar().showMessage(f"Audio saved to {file_path}")
            except Exception as e:
                logger.error(f"Error saving audio: {e}", exc_info=True)
                QMessageBox.warning(self, "Save Error", f"Error saving audio: {str(e)}")

    def load_history_item(self, item):
        """Load a history item when selected from the list"""
        idx = self.history_list.currentRow()
        if 0 <= idx < len(self.history):
            logger.debug(f"Loading history item at index {idx}")
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
            logger.debug(f"Playing history item at index {idx}")
            history_item = self.history[idx]
            
            # Save to temporary file for playback
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, history_item.audio, history_item.sr)
            
            # Play using simple audio player
            if self.audio_player.play(temp_file.name):
                # Enable stop button and update status
                self.stop_output_btn.setEnabled(True)
                self.statusBar().showMessage("Playing audio...")
                self.audio_player.current_temp_file = temp_file.name
                logger.info("Audio playback started from history item")
            else:
                logger.error("No suitable audio player found")
                QMessageBox.warning(self, "Playback Error", 
                                  "No suitable audio player found. Please install 'aplay', 'ffplay', or 'mplayer'.")

    def save_history_item(self):
        """Save the selected history item to a file"""
        idx = self.history_list.currentRow()
        if 0 <= idx < len(self.history):
            logger.debug(f"Saving history item at index {idx}")
            history_item = self.history[idx]
            
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self, "Save Audio", "", "WAV Files (*.wav);;MP3 Files (*.mp3);;All Files (*)"
            )
            
            if file_path:
                try:
                    # Determine format based on extension or selected filter
                    is_mp3 = file_path.lower().endswith('.mp3') or 'MP3' in selected_filter
                    
                    if is_mp3:
                        logger.info(f"Saving history item as MP3 to {file_path}")
                        # First save as temporary WAV
                        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        sf.write(temp_wav.name, history_item.audio, history_item.sr)
                        
                        # Convert to MP3 using pydub
                        audio = AudioSegment.from_wav(temp_wav.name)
                        audio.export(file_path, format="mp3")
                        
                        # Clean up temp file
                        os.unlink(temp_wav.name)
                    else:
                        # Save as WAV
                        logger.info(f"Saving history item to {file_path}")
                        sf.write(file_path, history_item.audio, history_item.sr)
                        
                    self.statusBar().showMessage(f"Audio saved to {file_path}")
                except Exception as e:
                    logger.error(f"Error saving history item: {e}", exc_info=True)
                    QMessageBox.warning(self, "Save Error", f"Error saving audio: {str(e)}")

    def delete_history_item(self):
        """Delete the selected history item"""
        idx = self.history_list.currentRow()
        if 0 <= idx < len(self.history):
            reply = QMessageBox.question(self, "Delete Item", 
                                       "Are you sure you want to delete this item?",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                logger.info(f"Deleting history item at index {idx}")
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
                logger.warning(f"Attempted to create duplicate profile: {name}")
                QMessageBox.warning(self, "New Profile", 
                                  f"A profile named '{name}' already exists.")
                return
                
            logger.info(f"Creating new profile: {name}")
            self.voice_profiles[name] = None
            self.profile_combo.addItem(name)
            self.profile_combo.setCurrentText(name)
            self.current_voice_profile = name
            
            self.statusBar().showMessage(f"Created new profile: {name}")

    def save_profile(self):
        """Save current embedding to the active profile"""
        if self.speaker_embedding is None:
            logger.warning("Attempted to save profile without embedding")
            QMessageBox.warning(self, "Save Profile", 
                              "Please analyze a voice sample first.")
            return
            
        # Save the embedding to current profile
        logger.info(f"Saving embedding to profile: {self.current_voice_profile}")
        
        # Handle different embedding types
        if hasattr(self.speaker_embedding, 'clone'):
            # For PyTorch tensors
            self.voice_profiles[self.current_voice_profile] = self.speaker_embedding.clone()
        elif isinstance(self.speaker_embedding, dict):
            # For dictionary-based embeddings (TTS models)
            self.voice_profiles[self.current_voice_profile] = self.speaker_embedding.copy()
        else:
            # For other types, just use as is
            self.voice_profiles[self.current_voice_profile] = self.speaker_embedding
        
        self.statusBar().showMessage(f"Voice saved to profile: {self.current_voice_profile}")

    def change_profile(self, profile_name):
        """Switch to a different voice profile"""
        if profile_name in self.voice_profiles:
            logger.info(f"Changing to profile: {profile_name}")
            self.current_voice_profile = profile_name
            embedding = self.voice_profiles[profile_name]
            
            if embedding is not None:
                self.speaker_embedding = embedding
                
                # Handle different embedding types
                if hasattr(embedding, 'clone'):
                    self.speaker_embedding = embedding.clone()
                elif isinstance(embedding, dict):
                    self.speaker_embedding = embedding.copy()
                
                self.generate_btn.setEnabled(True)
                self.save_profile_btn.setEnabled(True)
                self.statusBar().showMessage(f"Switched to profile: {profile_name}")
                logger.debug("Profile contains embedding - generation enabled")
            else:
                self.save_profile_btn.setEnabled(True)
                self.statusBar().showMessage(f"Selected empty profile: {profile_name}")
                logger.debug("Selected profile is empty")

    def closeEvent(self, event):
        """Clean up on close"""
        # Stop any playing audio
        logger.info("Application closing - cleaning up")
        self.audio_player.stop()
        event.accept()

# Main entry point
if __name__ == '__main__':
    print("Starting Voice Cloning App...")
    app = QApplication(sys.argv)
    window = ProductionVoiceCloningApp()
    window.show()
    print("Application window should be visible now")
    sys.exit(app.exec_())
