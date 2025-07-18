voice cloner
Voice Cloning Application
A PyQt5-based application for voice cloning and synthesis.
Setup Instructions

Install dependencies:
pip install -r requirements.txt

For audio playback, install one of these:
# On Ubuntu/Debian:
sudo apt-get install alsa-utils   # For aplay
sudo apt-get install ffmpeg       # For ffplay
sudo apt-get install mplayer      # For mplayer

# On macOS:
brew install sox                  # For play

test audio system
python test_audio.py


run application

python voice_cloning_app.py

## Simple XTTS Voice Cloner

For a minimal GUI that uses the open source [Coqui TTS](https://github.com/coqui-ai/TTS) `XTTS` voice cloning model run:

```bash
pip install TTS  # only required once
python simple_voice_cloner_gui.py
```

The application lets you pick a short voice sample, enter text and synthesize a WAV file using that voice.

