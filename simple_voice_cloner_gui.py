import os
from tkinter import Tk, Button, Label, Text, filedialog, messagebox

try:
    from TTS.api import TTS
except ImportError:
    raise ImportError("Please install the 'TTS' package: pip install TTS")


class SimpleVoiceCloner:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Voice Cloner")

        self.sample_path = None
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

        Button(root, text="Select Voice Sample", command=self.select_sample).pack(pady=5)

        Label(root, text="Enter text to synthesize:").pack()
        self.text_entry = Text(root, height=4, width=50)
        self.text_entry.pack(pady=5)

        self.clone_btn = Button(root, text="Clone Voice", command=self.clone_voice, state="disabled")
        self.clone_btn.pack(pady=5)

    def select_sample(self):
        path = filedialog.askopenfilename(title="Choose voice sample",
                                          filetypes=[("Audio files", "*.wav *.mp3"), ("All files", "*")])
        if path:
            self.sample_path = path
            self.clone_btn.config(state="normal")

    def clone_voice(self):
        text = self.text_entry.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("No Text", "Please enter text to synthesize.")
            return

        output = filedialog.asksaveasfilename(defaultextension=".wav",
                                              filetypes=[("WAV files", "*.wav")],
                                              title="Save synthesized audio")
        if not output:
            return

        try:
            self.model.tts_to_file(text=text, speaker_wav=self.sample_path, file_path=output)
            messagebox.showinfo("Done", f"Audio saved to {output}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


def main():
    root = Tk()
    app = SimpleVoiceCloner(root)
    root.mainloop()


if __name__ == "__main__":
    main()
