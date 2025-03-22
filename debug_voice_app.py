# Save this to debug_voice_app.py and run it
import sys
import os

# Add debug statements
print("Starting Voice Cloning App initialization...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

try:
    print("Importing PyQt5...")
    from PyQt5.QtWidgets import QApplication
    print("PyQt5 imported successfully")
    
    print("Importing NumPy...")
    import numpy as np
    print("NumPy imported successfully")
    
    print("Importing torch...")
    import torch
    print(f"PyTorch imported successfully, CUDA available: {torch.cuda.is_available()}")
    
    print("Importing librosa...")
    import librosa
    print("Librosa imported successfully")
    
    print("Importing soundfile...")
    import soundfile as sf
    print("Soundfile imported successfully")
    
    print("Importing matplotlib...")
    import matplotlib.pyplot as plt
    print("Matplotlib imported successfully")
    
    print("Importing PyAudio...")
    import pyaudio
    print("PyAudio imported successfully")
    
    # Import the main application class
    print("Importing main application class...")
    
    # This should be the name of your main class from the script
    from voice_cloning_app import ProductionVoiceCloningApp
    
    print("Starting application...")
    app = QApplication(sys.argv)
    window = ProductionVoiceCloningApp()
    window.show()
    print("Application window should be visible now")
    sys.exit(app.exec_())
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\nPlease make sure all dependencies are installed:")
    print("pip install PyQt5 numpy torch librosa soundfile pyaudio matplotlib")
