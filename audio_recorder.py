"""Simple microphone recorder utility."""

import queue
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write


class MicrophoneRecorder:
    def __init__(self, samplerate: int = 44100, channels: int = 1) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.recording = False
        self.stream = None
        self.q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(self, indata, frames, time, status) -> None:
        if status:
            print(status)
        self.q.put(indata.copy())

    def start_recording(self) -> None:
        self.recording = True
        self.q = queue.Queue()
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self._callback,
        )
        self.stream.start()

    def stop_recording(self) -> np.ndarray:
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.recording = False

        frames = []
        while not self.q.empty():
            frames.append(self.q.get())
        if frames:
            return np.concatenate(frames, axis=0)
        return np.array([], dtype=np.float32)

    def save_recording(self, path: str, data: np.ndarray) -> None:
        write(path, self.samplerate, data)
