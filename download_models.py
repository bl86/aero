#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from tqdm import tqdm

MODELS_DIR = Path.home() / "llm_system" / "models" / "llm"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RECOMMENDED_MODELS = {
    'coding': [
        ('deepseek-ai/deepseek-coder-6.7b-instruct', 'DeepSeek Coder 6.7B'),
        ('codellama/CodeLlama-13b-Instruct-hf', 'Code Llama 13B'),
    ],
    'general': [
        ('mistralai/Mistral-7B-Instruct-v0.2', 'Mistral 7B Instruct'),
        ('meta-llama/Llama-2-13b-chat-hf', 'Llama 2 13B Chat'),
    ],
    'multilingual': [
        ('google/gemma-7b-it', 'Gemma 7B'),
        ('tiiuae/falcon-7b-instruct', 'Falcon 7B Instruct'),
    ],
    'small': [
        ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'TinyLlama 1.1B'),
        ('microsoft/phi-2', 'Phi-2'),
    ]
}

def download_model(repo_id: str, model_name: str):
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Repository: {repo_id}")
    print(f"{'='*60}\n")

    local_dir = MODELS_DIR / repo_id.replace('/', '_')

    if local_dir.exists():
        print(f"✓ Model already exists at {local_dir}")
        return str(local_dir)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print(f"\n✓ Successfully downloaded to {local_dir}")
        return str(local_dir)

    except Exception as e:
        print(f"\n✗ Error downloading {model_name}: {str(e)}")
        return None

def show_menu():
    print("\n" + "="*60)
    print("Local LLM System - Model Downloader")
    print("="*60)

    print("\nRecommended Models by Category:\n")

    all_models = []
    idx = 1

    for category, models in RECOMMENDED_MODELS.items():
        print(f"\n{category.upper()}:")
        for repo_id, name in models:
            print(f"  {idx}. {name} ({repo_id})")
            all_models.append((repo_id, name))
            idx += 1

    print(f"\n  {idx}. Download all coding models")
    print(f"  {idx+1}. Download all small models (for testing)")
    print(f"  {idx+2}. Custom model (enter repo ID)")
    print(f"  0. Exit")

    return all_models

def download_category(category: str):
    models = RECOMMENDED_MODELS.get(category, [])

    for repo_id, name in models:
        download_model(repo_id, name)

def main():
    if not os.path.exists(Path.home() / "llm_system"):
        print("Error: LLM system not installed.")
        print("Please run install_llm.sh first.")
        sys.exit(1)

    while True:
        all_models = show_menu()

        try:
            choice = input("\nSelect option: ").strip()

            if choice == '0':
                print("Exiting...")
                break

            choice_num = int(choice)

            if 1 <= choice_num <= len(all_models):
                repo_id, name = all_models[choice_num - 1]
                download_model(repo_id, name)

            elif choice_num == len(all_models) + 1:
                print("\nDownloading all coding models...")
                download_category('coding')

            elif choice_num == len(all_models) + 2:
                print("\nDownloading all small models...")
                download_category('small')

            elif choice_num == len(all_models) + 3:
                repo_id = input("\nEnter HuggingFace repo ID (e.g., mistralai/Mistral-7B-v0.1): ").strip()
                if repo_id:
                    name = repo_id.split('/')[-1]
                    download_model(repo_id, name)

            else:
                print("Invalid option")

            input("\nPress Enter to continue...")

        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
