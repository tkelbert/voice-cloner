# Add these at the top of your voice_cloning_app_no_multimedia.py file:

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("voice_app_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VoiceApp")

# Then modify these methods to add logging:

def analyze_audio(self):
    """Analyze the source audio to extract speaker embedding"""
    if not hasattr(self, 'source_audio_path') or not self.source_audio_path:
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
        logger.debug(f"Embedding shape: {features['embedding'].shape}")
        logger.debug(f"Embedding device: {features['embedding'].device}")
        logger.debug(f"Embedding type: {features['embedding'].dtype}")
        
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
        logger.error(f"Error during audio playback: {str(e)}")
        QMessageBox.warning(self, "Playback Error", f"Error playing audio: {str(e)}")
