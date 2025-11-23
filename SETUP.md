# Local LLM System Setup Guide

Complete standalone AI system with GPU acceleration, multi-language support, and distributed training.

## System Requirements

### Hardware
- **Primary GPU**: RTX 3090 (24GB VRAM)
- **Remote GPUs** (optional): RTX 3080 rigs for distributed training
- **RAM**: 32GB+ recommended
- **Storage**: 100GB+ free space for models

### Software
- Ubuntu 20.04/22.04 or Debian-based Linux
- Python 3.10+
- CUDA-compatible NVIDIA drivers

## Installation

### Step 1: GPU Setup

For RTX 3090 with CUDA support:

```bash
chmod +x setup_gpu.sh
./setup_gpu.sh
sudo reboot
```

After reboot, verify:

```bash
nvidia-smi
nvcc --version
```

### Step 2: Install LLM System

```bash
chmod +x install_llm.sh
./install_llm.sh
```

This installs:
- vLLM for high-performance inference
- Whisper for speech-to-text
- Coqui TTS for text-to-speech
- Multi-language support (EN, SR, BS, HR)
- Agent framework
- Coding assistant tools
- Distributed training support

Installation directory: `~/llm_system/`

### Step 3: Download Models

```bash
source ~/llm_system/venv/bin/activate
python3 download_models.py
```

**Recommended models:**
- **For testing**: TinyLlama 1.1B (~2GB)
- **For coding**: DeepSeek Coder 6.7B (~13GB)
- **For general use**: Mistral 7B (~14GB)

Models are saved to: `~/llm_system/models/llm/`

## Quick Start

### Option 1: Interactive Script

```bash
chmod +x quick_start.sh
./quick_start.sh
```

### Option 2: Manual Launch

```bash
source ~/llm_system/venv/bin/activate

# Interactive mode
python3 -m system.main --mode interactive

# API server mode
python3 -m system.main --mode server --port 8000
```

## Features

### 1. Chat Interface

Interactive chat with the LLM:

```bash
python3 -m system.main --mode interactive
```

Commands:
- Type naturally to chat
- `/agent create <name>` - Create an agent
- `/code <task>` - Get coding help
- `/voice record` - Voice input
- `/train start` - Start training
- `help` - Show all commands

### 2. Multi-Language Support

Supported languages:
- English (en)
- Serbian (sr)
- Bosnian (bs)
- Croatian (hr)

The system auto-detects language or you can specify it explicitly.

### 3. Speech Capabilities

**Speech-to-Text** (using Whisper):
```python
from system.speech_engine import SpeechEngine
engine = SpeechEngine(config)

# Record and transcribe
text = engine.record_and_transcribe(language='sr')

# Transcribe file
text = engine.transcribe_file('audio.wav', language='hr')
```

**Text-to-Speech** (using Coqui TTS):
```python
# Generate speech
engine.text_to_speech("Hello, world!", language='en', play=True)

# Serbian
engine.text_to_speech("Здраво!", language='sr', play=True)
```

### 4. Agent System

Create and manage multiple agents:

```python
from system.agent_manager import AgentManager

# Create agent
agent = manager.create_agent("Assistant")

# Run task
response = manager.run_agent(agent.id, "Explain quantum computing")

# Specialized agents
coder = manager.create_specialized_agent('coder')
translator = manager.create_specialized_agent('translator')
```

### 5. Coding Assistant

Senior developer capabilities:

```python
from system.coding_assistant import CodingAssistant

assistant = CodingAssistant(config, llm_engine)

# Generate code
code = assistant.generate_code("Create a binary search tree", language='python')

# Review code
review = assistant.code_review(code, language='python')

# Debug code
fix = assistant.debug_code(buggy_code, error_message, language='python')

# Add tests
tests = assistant.add_tests(code, language='python')
```

### 6. Distributed Training

Train on multiple GPUs across machines:

```python
from system.distributed_trainer import DistributedTrainer

trainer = DistributedTrainer(config)

# Setup remote nodes (RTX 3080 rigs)
trainer.setup_remote_nodes([
    'user@192.168.1.100',
    'user@192.168.1.101'
])

# Start training
trainer.start_training(
    model_name='mistralai/Mistral-7B-v0.1',
    dataset_path='./data/training.json'
)

# Monitor
status = trainer.get_status()
gpus = trainer.monitor_gpus()
```

### 7. API Server

REST API for programmatic access:

```bash
python3 -m system.main --mode server --port 8000
```

API Documentation: `http://localhost:8000/docs`

**Endpoints:**

```bash
# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "language": "en"}'

# Generate code
curl -X POST http://localhost:8000/code/generate \
  -H "Content-Type: application/json" \
  -d '{"description": "Binary search algorithm", "language": "python"}'

# Create agent
curl -X POST http://localhost:8000/agent/create?name=Assistant

# List agents
curl http://localhost:8000/agent/list

# Speech-to-text
curl -X POST http://localhost:8000/speech/transcribe \
  -F "file=@audio.wav" \
  -F "language=sr"

# Text-to-speech
curl -X POST http://localhost:8000/speech/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "en"}'
```

## Configuration

Edit `~/llm_system/configs/system.yaml`:

```yaml
gpu:
  primary:
    device: cuda:0
    model: RTX_3090
    memory: 24576
  remote:
    enabled: true
    hosts:
      - user@192.168.1.100
      - user@192.168.1.101

llm:
  engine: vllm
  max_tokens: 8192
  temperature: 0.7
  gpu_memory_utilization: 0.9

speech:
  input:
    model: medium
    languages: [en, sr, bs, hr]
  output:
    languages: [en, sr, bs, hr]

languages:
  supported: [en, sr, bs, hr]
  default: en
  auto_detect: true

agents:
  max_concurrent: 5
  memory_enabled: true
```

## Remote GPU Setup

### On Remote Machines (RTX 3080)

1. Install GPU drivers and CUDA:
```bash
scp setup_gpu.sh user@remote-host:~/
ssh user@remote-host
./setup_gpu.sh
sudo reboot
```

2. Install Ray:
```bash
pip install ray[train] torch
```

3. Start Ray:
```bash
ray start --head --port=6379
```

### On Primary Machine (RTX 3090)

Configure remote hosts in `system.yaml` and the system will automatically distribute training workload.

## Performance Optimization

### RTX 3090 Settings

```bash
# Enable persistence mode
sudo nvidia-smi -pm 1

# Set power limit (350W)
sudo nvidia-smi -pl 350

# Set memory/GPU clocks
sudo nvidia-smi -ac 9751,1395
```

### Model Optimization

**For inference**:
- Use vLLM engine (default)
- FP16 precision
- GPU memory utilization: 0.9

**For training**:
- DeepSpeed ZeRO stage 2
- Gradient checkpointing
- Mixed precision (FP16)

## Usage Examples

### Example 1: Multilingual Voice Assistant

```python
from system.speech_engine import SpeechEngine
from system.llm_engine import LLMEngine

config = load_config()
speech = SpeechEngine(config)
llm = LLMEngine(config)

# Record in Serbian
print("Говорите...")
audio = speech.record_and_transcribe(language='sr')
print(f"Vi: {audio}")

# Generate response
response = llm.generate(audio)
print(f"AI: {response}")

# Speak response
speech.text_to_speech(response, language='sr')
```

### Example 2: Code Development Assistant

```python
from system.coding_assistant import CodingAssistant

assistant = CodingAssistant(config, llm_engine)

# Generate implementation
code = assistant.generate_code(
    "Create a REST API for user authentication with JWT",
    language='python'
)

# Review the code
review = assistant.code_review(code, language='python')

# Add tests
tests = assistant.add_tests(code, language='python')

# Create project
assistant.create_project('auth-api', 'python', 'api')
```

### Example 3: Distributed Model Training

```python
from system.distributed_trainer import DistributedTrainer

trainer = DistributedTrainer(config)

# Configure DeepSpeed
ds_config = trainer.configure_deepspeed({
    'train_batch_size': 32,
    'fp16': {'enabled': True},
    'zero_optimization': {'stage': 2}
})

# Train with LoRA
trainer.train_lora(
    base_model='mistralai/Mistral-7B-v0.1',
    dataset_path='./data/training.json',
    lora_config={
        'r': 8,
        'lora_alpha': 32,
        'target_modules': ['q_proj', 'v_proj']
    }
)
```

## Troubleshooting

### CUDA Out of Memory

1. Reduce `gpu_memory_utilization` in config
2. Use smaller batch size
3. Enable gradient checkpointing
4. Use model quantization

### Speech Engine Issues

1. Install audio dependencies:
```bash
sudo apt-get install portaudio19-dev espeak-ng
```

2. Check microphone:
```bash
arecord -l
```

### Model Download Failures

1. Check internet connection
2. Verify HuggingFace token (if needed):
```bash
huggingface-cli login
```

3. Resume download:
```bash
python3 download_models.py
```

## Directory Structure

```
~/llm_system/
├── venv/              # Python environment
├── models/
│   ├── llm/          # Language models
│   └── speech/       # Speech models
├── configs/          # Configuration files
├── cache/            # Model cache
├── logs/             # System logs
├── agents/           # Agent data
├── workspace/        # Coding workspace
└── training/         # Training checkpoints
```

## System Information

All components are local and standalone:
- No external API calls
- No data transmission to cloud services
- Complete privacy and control
- Runs entirely on your hardware

For support, check logs in `~/llm_system/logs/`
