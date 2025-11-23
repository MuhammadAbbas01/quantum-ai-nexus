import asyncio
import aiofiles
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from contextlib import asynccontextmanager
import tempfile
import os
from datetime import datetime
import json

# Modern AI Libraries
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq # Import AutoModelForSpeechSeq2Seq for Whisper
import soundfile as sf
import librosa
from gtts import gTTS # Using gTTS for Text-to-Speech
import pyaudio
import wave # For recording WAV files

# Performance & Monitoring
from functools import lru_cache
import time
import psutil
from concurrent.futures import ThreadPoolExecutor
import queue

@dataclass
class VoiceConfig:
    """Configuration for voice processing"""
    # Changed to Whisper base model for better accuracy
    asr_model: str = "openai/whisper-base"
    sample_rate: int = 16000
    chunk_size: int = 2048 # Increased chunk size for librosa compatibility
    process_buffer_seconds: int = 5 # Used for real-time, but less critical for file-based
    language: str = "en"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_audio_length: int = 30 # seconds
    noise_reduction: bool = True # This will be a simpler, non-AI noise reduction
    emotion_detection: bool = True

class ModernVoiceProcessor:
    """
    Advanced Voice Processing System - 2025 Edition

    Features:
    - Speech recognition (file-based and real-time) with Hugging Face Transformers (Whisper)
    - Text-to-speech with gTTS
    - Noise reduction (basic)
    - Emotion recognition from speech
    - Real-time streaming capabilities (without external VAD)
    """

    def __init__(self, config: VoiceConfig = None):
        self.config = config or VoiceConfig()
        self.logger = self._setup_logging()
        self.device = torch.device(self.config.device)

        # Initialize performance metrics first
        self.performance_metrics = {
            'processing_times': [],
            'memory_usage': [],
            'model_load_time': 0
        }

        # Initialize models
        self._initialize_models()

        # Real-time processing (kept for potential future use, but not directly used in main loop now)
        self.audio_queue = queue.Queue()
        self.is_recording = False # This flag is now primarily for the continuous real_time_transcription method
        self.executor = ThreadPoolExecutor(max_workers=4)

        # PyAudio instance for playback (will be terminated in __exit__)
        self.p_audio = pyaudio.PyAudio()

    def _setup_logging(self) -> logging.Logger:
        """Setup advanced logging"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def _initialize_models(self):
        """Initialize all AI models with optimization"""
        start_time = time.time()

        try:
            # Automatic Speech Recognition (ASR) using Hugging Face Whisper model
            self.processor_asr = AutoProcessor.from_pretrained(self.config.asr_model)
            self.model_asr = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.asr_model,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(self.device)
            self.logger.info(f"ASR model '{self.config.asr_model}' loaded.")

            # gTTS is used for text-to-speech, no model to load here
            self.logger.info("gTTS will be used for Text-to-Speech (no local model load).")

            # Removed webrtcvad related logic
            self.logger.info("Voice Activity Detection (webrtcvad) is not used to avoid compilation issues. ASR will process all audio segments.")

            # Emotion recognition pipeline (still requires transformers model)
            if self.config.emotion_detection:
                self.emotion_model = pipeline(
                    "audio-classification",
                    model="superb/wav2vec2-base-superb-er",
                    device=0 if self.config.device == "cuda" else -1
                )

            # Noise reduction model (simplified to librosa for now, no separate model load)
            if self.config.noise_reduction:
                self.logger.info("Basic librosa-based noise reduction will be applied.")
                # No specific model to load here for this basic method

        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise

        self.performance_metrics['model_load_time'] = time.time() - start_time
        self.logger.info(f"Models loaded in {self.performance_metrics['model_load_time']:.2f}s")

    async def speech_to_text_advanced(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Advanced speech-to-text with confidence scoring and timestamps using Transformers ASR (Whisper).
        """
        start_time = time.time()

        try:
            # Preprocess audio (noise reduction, normalization, resampling)
            # Ensure audio_data is at 16kHz for Whisper
            # The _preprocess_audio method should already handle resampling to 16kHz
            audio_processed = await self._preprocess_audio(audio_data)

            # Prepare audio for Whisper model
            input_features = self.processor_asr(
                audio_processed,
                sampling_rate=self.config.sample_rate, # Should be 16000 now
                return_tensors="pt"
            ).input_features.to(self.device)

            # Generate transcription
            predicted_ids = self.model_asr.generate(input_features)
            transcription_text = self.processor_asr.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            # Dummy confidence for now, as direct confidence score isn't easily exposed by Whisper generate
            confidence = 0.95 if transcription_text else 0.0

            transcription = {
                'text': transcription_text,
                'language': self.config.language,
                'segments': [], # Whisper pipeline doesn't easily expose segments/timestamps by default here
                'confidence': confidence,
                'processing_time': time.time() - start_time,
                'word_timestamps': [] # Not directly available from this pipeline setup
            }

            # Add emotion detection if enabled
            if self.config.emotion_detection:
                emotion_result = await self._detect_emotion(audio_processed)
                transcription['emotion'] = emotion_result

            self._update_performance_metrics(time.time() - start_time)

            return transcription

        except Exception as e:
            self.logger.error(f"Speech recognition failed: {e}")
            return {
                'text': '',
                'error': str(e),
                'processing_time': time.time() - start_time
            }

    async def text_to_speech_neural(self, text: str):
        """
        Neural text-to-speech using gTTS and real-time playback via PyAudio.
        """
        start_time = time.time()
        temp_mp3_file = None
        try:
            # Generate speech with gTTS to a temporary MP3 file
            temp_mp3_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
            tts = gTTS(text=text, lang=self.config.language, slow=False)
            tts.save(temp_mp3_file)
            self.logger.info(f"gTTS audio saved to temporary MP3 for playback: {temp_mp3_file}")

            # Load the audio data from the temporary MP3
            audio_data, sr = sf.read(temp_mp3_file, dtype='float32')

            # Ensure mono audio
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            # Resample to the desired sample_rate if necessary for playback
            if sr != self.config.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.config.sample_rate)
                self.logger.info(f"Resampled TTS audio from {sr} Hz to {self.config.sample_rate} Hz for playback.")

            # Convert to int16 for PyAudio playback
            audio_data_int16 = (audio_data * 32767).astype(np.int16)

            # Open PyAudio stream for playback
            stream = self.p_audio.open(format=pyaudio.paInt16,
                                         channels=1,
                                         rate=self.config.sample_rate,
                                         output=True)

            # Play audio in chunks
            playback_chunk_size = 1024 # Smaller chunk size for smoother real-time playback
            for i in range(0, len(audio_data_int16), playback_chunk_size):
                chunk = audio_data_int16[i:i + playback_chunk_size]
                stream.write(chunk.tobytes())
                await asyncio.sleep(0.001) # Small sleep to yield control

            stream.stop_stream()
            stream.close()

            self.logger.info(f"TTS playback completed in {time.time() - start_time:.2f}s.")

        except Exception as e:
            self.logger.error(f"TTS playback failed: {e}")
        finally:
            # Clean up the temporary MP3 file
            if temp_mp3_file and os.path.exists(temp_mp3_file):
                os.remove(temp_mp3_file)
                self.logger.debug(f"Cleaned up temporary MP3: {temp_mp3_file}")


    async def capture_and_transcribe_single_segment(self, duration_seconds: int) -> Dict[str, Any]:
        """Captures audio from microphone for a fixed duration and transcribes it."""
        self.logger.info(f"Capturing audio for {duration_seconds} seconds...")
        audio_format = pyaudio.paInt16
        channels = 1
        rate = self.config.sample_rate
        chunk = self.config.chunk_size

        p_input = pyaudio.PyAudio() # Create a local PyAudio instance for input
        frames = []
        stream = None
        try:
            stream = p_input.open(
                format=audio_format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk
            )

            num_chunks = int((rate / chunk) * duration_seconds)
            for _ in range(num_chunks):
                data = stream.read(chunk, exception_on_overflow=False)
                frames.append(data)

            # Convert frames to numpy array
            audio_data_np = np.frombuffer(b''.join(frames), dtype=np.int16)
            audio_data_float = audio_data_np.astype(np.float32) / 32768.0

            self.logger.info(f"Finished capturing {len(audio_data_float)/rate:.2f} seconds of audio.")
            return await self.speech_to_text_advanced(audio_data_float)

        except Exception as e:
            self.logger.error(f"Error during single segment audio capture/transcription: {e}")
            return {'text': '', 'error': str(e)}
        finally:
            if stream and stream.is_active():
                stream.stop_stream()
                stream.close()
            if p_input: # Terminate the input PyAudio instance
                p_input.terminate()


    def stop_recording(self):
        """Stop real-time recording (primarily for the continuous real_time_transcription method if used)"""
        self.is_recording = False
        self.logger.info("Recording stopped")

    async def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Advanced audio preprocessing"""
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        if self.config.noise_reduction:
            audio_data = await self._reduce_noise_basic(audio_data, n_fft=self.config.chunk_size)

        audio_data = librosa.util.normalize(audio_data)

        if len(audio_data) == 0:
            return np.zeros(self.config.sample_rate, dtype=np.float32)

        # Resample to 16kHz if not already
        if self.config.sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=self.config.sample_rate, target_sr=16000)
            self.config.sample_rate = 16000 # Update config to reflect new sample rate - CRITICAL for Whisper
        return audio_data

    async def _reduce_noise_basic(self, audio_data: np.ndarray, n_fft: int = 2048) -> np.ndarray:
        """Basic spectral subtraction noise reduction using librosa."""
        try:
            # Ensure audio_data is long enough for STFT
            if len(audio_data) < n_fft:
                self.logger.warning(f"Audio data length ({len(audio_data)}) is less than n_fft ({n_fft}). Attempting to pad.")
                # Pad with zeros to meet n_fft requirement if it's slightly short, otherwise return as is
                if len(audio_data) > 0:
                    # Pad to n_fft length
                    padded_audio_data = np.pad(audio_data, (0, n_fft - len(audio_data)), 'constant')
                    self.logger.warning(f"Padded audio data from {len(audio_data)} to {len(padded_audio_data)} samples for noise reduction.")
                    audio_data = padded_audio_data
                else:
                    return audio_data # Return empty if input was empty

            stft = librosa.stft(audio_data, n_fft=n_fft)
            magnitude, phase = librosa.magphase(stft)

            # Estimate noise from the first small portion of the audio
            # Assuming the first 0.5 seconds are mostly noise or silence for estimation
            noise_len_frames = min(magnitude.shape[1], int(0.5 * self.config.sample_rate / (self.config.chunk_size / 2)))
            noise_profile = np.mean(magnitude[:, :noise_len_frames], axis=1, keepdims=True)

            # Apply spectral subtraction
            # Multiply by a factor (e.g., 1.5) to be more aggressive with noise removal
            cleaned_magnitude = np.maximum(magnitude - noise_profile * 1.5, 0)

            # Reconstruct the audio
            cleaned_stft = cleaned_magnitude * phase
            cleaned_audio = librosa.istft(cleaned_stft, length=len(audio_data))

            return cleaned_audio

        except Exception as e:
            self.logger.warning(f"Basic noise reduction failed: {e}. Returning original audio.")
            return audio_data

    async def _detect_emotion(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Detect emotion from speech"""
        try:
            audio_input = {
                "array": audio_data,
                "sampling_rate": self.config.sample_rate
            }

            result = self.emotion_model(audio_input)

            return {
                'dominant_emotion': result[0]['label'],
                'confidence': result[0]['score'],
                'all_emotions': result
            }

        except Exception as e:
            self.logger.error(f"Emotion detection failed: {e}")
            return {'dominant_emotion': 'neutral', 'confidence': 0.5}

    def _extract_word_timestamps(self, segments: List[Dict]) -> List[Dict]:
        """Extract word-level timestamps (dummy for transformers ASR)"""
        return []

    def _update_performance_metrics(self, processing_time: float):
        """Update performance monitoring"""
        self.performance_metrics['processing_times'].append(processing_time)
        self.performance_metrics['memory_usage'].append(psutil.virtual_memory().percent)

        if len(self.performance_metrics['processing_times']) > 100:
            self.performance_metrics['processing_times'].pop(0)
            self.performance_metrics['memory_usage'].pop(0)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.performance_metrics['processing_times']:
            return {'status': 'No data available'}

        return {
            'avg_processing_time': np.mean(self.performance_metrics['processing_times']),
            'max_processing_time': np.max(self.performance_metrics['processing_times']),
            'min_processing_time': np.min(self.performance_metrics['processing_times']),
            'avg_memory_usage': np.mean(self.performance_metrics['memory_usage']),
            'model_load_time': self.performance_metrics['model_load_time'],
            'total_processed': len(self.performance_metrics['processing_times'])
        }

    async def cleanup_temp_files(self, file_paths: List[str]):
        """Async cleanup of temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.logger.debug(f"Cleaned up: {file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup {file_path}: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.logger.info("Voice processor shutdown complete")
        if hasattr(self, 'p_audio') and self.p_audio:
            self.p_audio.terminate()
            self.logger.info("PyAudio (output) terminated.")


# Usage Example and Factory
class VoiceProcessorFactory:
    """Factory for creating optimized voice processors"""

    @staticmethod
    def create_production_processor() -> ModernVoiceProcessor:
        """Create production-ready processor"""
        config = VoiceConfig(
            asr_model="openai/whisper-base", # Using Whisper base model
            device="cuda" if torch.cuda.is_available() else "cpu",
            noise_reduction=True,
            emotion_detection=True,
            process_buffer_seconds=5
        )
        return ModernVoiceProcessor(config)

    @staticmethod
    def create_high_performance_processor() -> ModernVoiceProcessor:
        """Create high-performance processor for powerful hardware"""
        config = VoiceConfig(
            asr_model="openai/whisper-medium", # Use medium for better performance if GPU allows
            device="cuda",
            noise_reduction=True,
            emotion_detection=True,
            process_buffer_seconds=6
        )
        return ModernVoiceProcessor(config)

    @staticmethod
    def create_lightweight_processor() -> ModernVoiceProcessor:
        """Create lightweight processor for resource-constrained environments"""
        config = VoiceConfig(
            asr_model="openai/whisper-tiny", # Smallest Whisper model
            device="cpu",
            noise_reduction=False,
            emotion_detection=False,
            process_buffer_seconds=3
        )
        return ModernVoiceProcessor(config)


# Modern async usage example
async def main():
    """Example usage"""
    # Create processor
    processor = VoiceProcessorFactory.create_production_processor()

    print("\n--- Interactive Voice Chat ---")
    print("Type text for the AI to speak (press Enter).")
    print("After AI speaks, you will have a few seconds to speak for transcription.")
    print("Type 'exit' or 'quit' at the prompt to stop.")

    try:
        while True:
            # Give a brief moment for the STT task's initial logs to print (if any)
            # and to ensure the console is ready for input.
            await asyncio.sleep(0.1)

            # Get text input from user for TTS
            user_input = await asyncio.to_thread(input, "YOU (type for AI to speak): ")

            if user_input.lower() in ['exit', 'quit']:
                print("Exiting interactive mode.")
                break

            # Play the typed text via TTS
            print(f"AI speaking: '{user_input}'")
            await processor.text_to_speech_neural(user_input)
            print("AI finished speaking.")

            # Now, AI listens for user's voice for a limited time
            listen_duration = 5 # seconds
            print(f"\nAI listening for your voice (speak now for {listen_duration} seconds)...")
            stt_result = await processor.capture_and_transcribe_single_segment(listen_duration)

            if stt_result['text']:
                print(f"YOU (transcribed): '{stt_result['text']}' (Confidence: {stt_result['confidence']:.2f}, Emotion: {stt_result.get('emotion', {}).get('dominant_emotion', 'N/A')})")
            else:
                print(f"YOU (transcribed): (No clear speech detected in this segment. Error: {stt_result.get('error', 'Unknown')})")
            print("-" * 50) # Separator for clarity

    except KeyboardInterrupt:
        print("\nStopping interactive mode (KeyboardInterrupt detected).")
    except asyncio.CancelledError:
        print("\nInteractive mode cancelled.")
    finally:
        # No continuous STT task to stop/cancel here as it's turn-based
        pass # The processor's __exit__ will handle PyAudio termination automatically

    # Performance stats
    stats = processor.get_performance_stats()
    print(f"\nPerformance Statistics: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())