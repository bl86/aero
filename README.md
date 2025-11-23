# Local LLM System

Complete standalone AI system with GPU acceleration, multi-language support, voice capabilities, and distributed training.

## Features

- **Standalone & Private**: No external API calls, complete local control
- **GPU Accelerated**: Optimized for RTX 3090, supports distributed training on RTX 3080 rigs
- **Multi-language**: English, Serbian, Bosnian, Croatian support
- **Voice Capable**: Speech-to-text and text-to-speech in multiple languages
- **Agent System**: Create and manage multiple specialized AI agents
- **Coding Assistant**: Senior developer-level coding help
- **Distributed Training**: Train models across multiple GPUs and machines

## Quick Start

### 1. Install System

```bash
chmod +x quick_start.sh
./quick_start.sh
```

This will:
- Install GPU drivers (if needed)
- Set up the LLM system
- Download models
- Configure the environment

### 2. Start Using

```bash
source ~/llm_system/venv/bin/activate
python3 -m system.main --mode interactive
```

## System Requirements

- **GPU**: RTX 3090 (24GB VRAM recommended)
- **RAM**: 32GB+
- **Storage**: 100GB+ for models
- **OS**: Ubuntu 20.04/22.04 or Debian-based Linux

## Installation Options

### Option 1: Automated (Recommended)

```bash
./quick_start.sh
```

### Option 2: Manual

```bash
# 1. Setup GPU
chmod +x setup_gpu.sh
./setup_gpu.sh
sudo reboot

# 2. Install system
chmod +x install_llm.sh
./install_llm.sh

# 3. Download models
source ~/llm_system/venv/bin/activate
python3 download_models.py
```

## Usage Examples

### Interactive Chat

```bash
python3 -m system.main --mode interactive
```

### Voice Chat (Multi-language)

```bash
python3 examples/voice_chat.py
```

### Coding Assistant

```bash
python3 examples/code_assistant.py
```

### Multi-language Agents

```bash
python3 examples/multilang_agent.py
```

### API Server

```bash
python3 -m system.main --mode server --port 8000
```

Access API docs at: `http://localhost:8000/docs`

## Configuration

Edit `~/llm_system/configs/system.yaml` to customize:

- GPU settings
- Model parameters
- Language support
- Agent configuration
- Remote GPU hosts

## Remote GPU Setup

To use remote RTX 3080 rigs for distributed training:

1. Copy setup to remote machines:
```bash
scp setup_gpu.sh user@remote:~/
```

2. Install on remote machines:
```bash
ssh user@remote
./setup_gpu.sh
sudo reboot
```

3. Configure in `system.yaml`:
```yaml
gpu:
  remote:
    enabled: true
    hosts:
      - user@192.168.1.100
      - user@192.168.1.101
```

## System Management

```bash
python3 utils/system_manager.py
```

Features:
- Monitor GPU/CPU/RAM usage
- Check installed models
- Clean cache
- Backup configurations
- View logs

## Documentation

- [Complete Setup Guide](SETUP.md)
- Examples in `examples/` directory
- Utilities in `utils/` directory

## Architecture

```
~/llm_system/
├── venv/              # Python environment
├── models/            # AI models
│   ├── llm/          # Language models
│   └── speech/       # Speech models
├── configs/          # Configuration
├── cache/            # Temporary cache
├── logs/             # System logs
├── agents/           # Agent data
├── workspace/        # Coding workspace
└── training/         # Training data
```

## Components

- **LLM Engine**: High-performance inference with vLLM
- **Speech Engine**: Multi-language STT/TTS
- **Agent Manager**: Create and manage AI agents
- **Coding Assistant**: Code generation, review, debugging
- **Distributed Trainer**: Multi-GPU training across machines
- **API Server**: REST API for programmatic access

## Models

Recommended models for RTX 3090:

- **Coding**: DeepSeek Coder 6.7B, Code Llama 13B
- **General**: Mistral 7B, Llama 2 13B
- **Testing**: TinyLlama 1.1B, Phi-2

Download via:
```bash
python3 download_models.py
```

## Performance

RTX 3090 optimizations:
- vLLM for inference (3-5x faster)
- FP16 precision
- 90% GPU memory utilization
- DeepSpeed for training

Distributed training across multiple RTX 3080s supported via Ray.

## Privacy

- All processing is local
- No data sent to external services
- No telemetry or tracking
- Complete control over your data

## Support

Check `~/llm_system/logs/` for debugging information.

## License

MIT

---

**Note**: This is a standalone system. All components run locally on your hardware. No external dependencies or API keys required.
