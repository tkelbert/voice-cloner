#!/usr/bin/env python3
# test_audio.py - Simple audio system test

import numpy as np
import soundfile as sf
import subprocess
import os

print("Audio System Test")
print("=================")

# Create a simple beep sound
print("Creating test audio file...")
sample_rate = 22050
duration = 1  # seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
beep = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

# Save to a file
test_file = "test_beep.wav"
sf.write(test_file, beep, sample_rate)
print(f"Created {test_file}")

# Try different players
players = [
    ["aplay", test_file],
    ["ffplay", "-nodisp", "-autoexit", test_file],
    ["mplayer", test_file],
    ["mpg123", test_file],
    ["paplay", test_file],
    ["play", test_file]
]

print("\nTesting audio players:")
success = False

for player_cmd in players:
    player_name = player_cmd[0]
    
    # Check if command exists
    try:
        which_process = subprocess.run(["which", player_name], 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE)
        exists = which_process.returncode == 0
        
        if exists:
            print(f"Found {player_name}, attempting to play test audio...")
            
            # Try to play
            try:
                process = subprocess.run(player_cmd, 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE,
                                       timeout=3)
                
                if process.returncode == 0:
                    print(f"SUCCESS: {player_name} played the audio file")
                    success = True
                else:
                    print(f"FAILED: {player_name} returned error code {process.returncode}")
                    if process.stderr:
                        error = process.stderr.decode('utf-8', errors='ignore')
                        print(f"Error message: {error}")
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT: {player_name} might be still playing or hung")
            except Exception as e:
                print(f"ERROR: {player_name} failed with exception: {str(e)}")
        else:
            print(f"{player_name} not found")
    except Exception as e:
        print(f"Error checking for {player_name}: {str(e)}")

print("\nAudio test summary:")
if success:
    print("At least one audio player worked! Your system can play audio.")
else:
    print("No audio players worked. You need to install audio playback software.")
    print("Try: sudo apt-get install alsa-utils ffmpeg mplayer")

# Clean up
try:
    os.remove(test_file)
    print(f"Removed test file {test_file}")
except:
    print(f"Could not remove test file {test_file}")
