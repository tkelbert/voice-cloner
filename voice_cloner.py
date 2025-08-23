import torch
from TTS.api import TTS


class VoiceCloner:
    """Wrapper around Coqui TTS XTTSv2 model."""

    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model_name)
        self.tts.to(device)
        self.device = device

    def synthesize(self, text: str, speaker_wav: str, language: str = "en"):
        """Synthesize speech and return the audio array and sample rate."""
        audio = self.tts.tts(text=text, speaker_wav=speaker_wav, language=language)
        sr = self.tts.synthesizer.output_sample_rate
        return audio, sr

    def synthesize_to_file(self, text: str, speaker_wav: str, file_path: str, language: str = "en") -> None:
        """Synthesize speech and save it to ``file_path``."""
        self.tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=file_path)
