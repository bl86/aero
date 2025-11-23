#!/usr/bin/env python3

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path.home() / "llm_system"))

from system.speech_engine import SpeechEngine
from system.llm_engine import LLMEngine

def load_config():
    config_path = Path.home() / "llm_system" / "configs" / "system.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("Initializing Voice Chat System...")
    print("="*50)

    config = load_config()

    speech = SpeechEngine(config)
    llm = LLMEngine(config)

    print("\nSupported languages: English, Serbian, Bosnian, Croatian")
    print("The system will auto-detect your language.\n")

    print("Voice Chat Active!")
    print("Speak after each prompt. Press Ctrl+C to exit.\n")

    conversation_history = []

    try:
        while True:
            print("\n[Listening... Speak now]")

            audio = speech.record_until_silence()

            detected_lang = speech.detect_language(audio)
            print(f"Detected language: {detected_lang}")

            text = speech.transcribe(audio, language=detected_lang)
            print(f"\nYou: {text}")

            if text.lower() in ['exit', 'quit', 'stop']:
                print("Goodbye!")
                break

            conversation_history.append({
                'role': 'user',
                'content': text
            })

            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

            response = llm.chat(conversation_history)

            conversation_history.append({
                'role': 'assistant',
                'content': response
            })

            print(f"\nAssistant: {response}\n")

            print("[Speaking response...]")
            speech.text_to_speech(response, language=detected_lang)

    except KeyboardInterrupt:
        print("\n\nExiting voice chat. Goodbye!")

if __name__ == "__main__":
    main()
