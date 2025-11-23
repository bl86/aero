import os
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from typing import Optional, List
from faster_whisper import WhisperModel
from TTS.api import TTS
import io
import wave

class SpeechEngine:
    def __init__(self, config):
        self.config = config
        self.base_dir = Path(config['system']['install_dir']).expanduser()
        self.models_dir = self.base_dir / "models" / "speech"
        self.cache_dir = self.base_dir / "cache"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.stt_model = None
        self.tts_model = None

        self.supported_languages = config['speech']['input']['languages']

        self.language_codes = {
            'en': 'en',
            'sr': 'sr',
            'bs': 'bs',
            'hr': 'hr'
        }

        self._init_stt()
        self._init_tts()

    def _init_stt(self):
        model_size = self.config['speech']['input'].get('model', 'medium')

        compute_type = "float16" if self.device == "cuda" else "int8"

        self.stt_model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=compute_type
        )

    def _init_tts(self):
        tts_models = {
            'en': 'tts_models/en/ljspeech/tacotron2-DDC',
            'sr': 'tts_models/multilingual/multi-dataset/your_tts',
            'bs': 'tts_models/multilingual/multi-dataset/your_tts',
            'hr': 'tts_models/multilingual/multi-dataset/your_tts'
        }

        self.tts_models = {}
        self.default_tts = TTS(
            model_name='tts_models/multilingual/multi-dataset/your_tts',
            gpu=(self.device == "cuda")
        )

    def record_audio(self, duration: int = 5, sample_rate: int = 16000) -> np.ndarray:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        return recording.flatten()

    def record_until_silence(self, sample_rate: int = 16000,
                            silence_threshold: float = 0.01,
                            silence_duration: float = 1.5) -> np.ndarray:

        import queue
        import threading

        audio_queue = queue.Queue()
        recording = []
        is_recording = [True]

        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            audio_queue.put(indata.copy())

        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            callback=audio_callback
        )

        with stream:
            silence_frames = 0
            silence_threshold_frames = int(silence_duration * sample_rate / 1024)

            while is_recording[0]:
                try:
                    data = audio_queue.get(timeout=0.1)
                    recording.append(data)

                    if np.abs(data).mean() < silence_threshold:
                        silence_frames += 1
                        if silence_frames >= silence_threshold_frames:
                            break
                    else:
                        silence_frames = 0

                except queue.Empty:
                    continue

        audio = np.concatenate(recording, axis=0).flatten()
        return audio

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None,
                  sample_rate: int = 16000) -> str:

        audio_file = io.BytesIO()
        sf.write(audio_file, audio, sample_rate, format='WAV')
        audio_file.seek(0)

        temp_path = self.cache_dir / "temp_audio.wav"
        with open(temp_path, 'wb') as f:
            f.write(audio_file.read())

        if language and language in self.language_codes:
            lang_code = self.language_codes[language]
        else:
            lang_code = None

        segments, info = self.stt_model.transcribe(
            str(temp_path),
            language=lang_code,
            beam_size=5
        )

        transcription = " ".join([segment.text for segment in segments])

        os.remove(temp_path)

        return transcription.strip()

    def record_and_transcribe(self, duration: int = None,
                             language: Optional[str] = None) -> str:

        if duration:
            audio = self.record_audio(duration)
        else:
            print("Recording... (will stop after silence)")
            audio = self.record_until_silence()

        return self.transcribe(audio, language)

    def text_to_speech(self, text: str, language: str = 'en',
                      output_file: Optional[str] = None,
                      play: bool = True) -> Optional[str]:

        if language not in self.supported_languages:
            language = 'en'

        if output_file is None:
            output_file = str(self.cache_dir / "temp_tts.wav")

        if language == 'en':
            self.default_tts.tts_to_file(
                text=text,
                file_path=output_file,
                language='en'
            )
        else:
            lang_map = {
                'sr': 'sr',
                'bs': 'hr',
                'hr': 'hr'
            }

            tts_lang = lang_map.get(language, 'en')

            self.default_tts.tts_to_file(
                text=text,
                file_path=output_file,
                language=tts_lang
            )

        if play:
            self.play_audio(output_file)

        return output_file

    def play_audio(self, file_path: str):
        data, sample_rate = sf.read(file_path)
        sd.play(data, sample_rate)
        sd.wait()

    def detect_language(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        temp_path = self.cache_dir / "temp_detect.wav"
        sf.write(temp_path, audio, sample_rate)

        segments, info = self.stt_model.transcribe(
            str(temp_path),
            language=None
        )

        os.remove(temp_path)

        return info.language

    def save_audio(self, audio: np.ndarray, file_path: str,
                  sample_rate: int = 16000):
        sf.write(file_path, audio, sample_rate)

    def load_audio(self, file_path: str) -> tuple:
        data, sample_rate = sf.read(file_path)
        return data, sample_rate

    def transcribe_file(self, file_path: str, language: Optional[str] = None) -> str:
        audio, sample_rate = self.load_audio(file_path)
        return self.transcribe(audio, language, sample_rate)

    def get_supported_languages(self) -> List[str]:
        return self.supported_languages

    def batch_transcribe(self, audio_files: List[str],
                        language: Optional[str] = None) -> List[str]:
        results = []
        for file_path in audio_files:
            try:
                text = self.transcribe_file(file_path, language)
                results.append(text)
            except Exception as e:
                results.append(f"Error: {str(e)}")

        return results

    def conversation_mode(self, language: str = None):
        print("Conversation mode activated. Press Ctrl+C to exit.")
        print("Speak after the prompt...\n")

        try:
            while True:
                print("\n[Listening...]")
                audio = self.record_until_silence()

                if language is None:
                    detected_lang = self.detect_language(audio)
                    print(f"Detected language: {detected_lang}")
                    use_lang = detected_lang
                else:
                    use_lang = language

                text = self.transcribe(audio, use_lang)
                print(f"You: {text}")

                yield text

        except KeyboardInterrupt:
            print("\nExiting conversation mode.")
