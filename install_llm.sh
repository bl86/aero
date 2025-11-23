#!/bin/bash

set -e

INSTALL_DIR="$HOME/llm_system"
MODELS_DIR="$INSTALL_DIR/models"
VENV_DIR="$INSTALL_DIR/venv"

echo "Local LLM System Installation"
echo "=============================="

create_directories() {
    echo "Creating directory structure..."
    mkdir -p $INSTALL_DIR
    mkdir -p $MODELS_DIR/{llm,speech,embeddings}
    mkdir -p $INSTALL_DIR/{logs,cache,data,configs}
}

install_system_deps() {
    echo "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        python3.10 \
        python3.10-venv \
        python3-pip \
        ffmpeg \
        libsndfile1 \
        portaudio19-dev \
        espeak-ng \
        git-lfs \
        libopenblas-dev \
        build-essential \
        cmake \
        pkg-config
}

setup_python_env() {
    echo "Setting up Python environment..."
    python3.10 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate

    pip install --upgrade pip setuptools wheel
}

install_llm_engines() {
    echo "Installing LLM engines..."
    source $VENV_DIR/bin/activate

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    pip install \
        vllm \
        transformers \
        accelerate \
        bitsandbytes \
        sentencepiece \
        protobuf \
        einops \
        tiktoken \
        peft \
        trl

    pip install llama-cpp-python \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
}

install_speech_engines() {
    echo "Installing speech processing engines..."
    source $VENV_DIR/bin/activate

    pip install \
        openai-whisper \
        faster-whisper \
        TTS \
        soundfile \
        librosa \
        pyaudio \
        pydub \
        sounddevice

    pip install noisereduce webrtcvad
}

install_multilang_support() {
    echo "Installing multi-language support..."
    source $VENV_DIR/bin/activate

    pip install \
        langdetect \
        polyglot \
        pyicu \
        pycld2 \
        morfessor
}

install_agent_framework() {
    echo "Installing agent framework..."
    source $VENV_DIR/bin/activate

    pip install \
        langchain \
        langchain-community \
        chromadb \
        faiss-cpu \
        sentence-transformers \
        instructor \
        outlines
}

install_dev_tools() {
    echo "Installing development tools..."
    source $VENV_DIR/bin/activate

    pip install \
        code-interpreter \
        tree-sitter \
        tree-sitter-languages \
        pygments \
        black \
        pylint \
        mypy \
        pytest \
        ipython
}

install_distributed_training() {
    echo "Installing distributed training tools..."
    source $VENV_DIR/bin/activate

    pip install \
        ray[train] \
        deepspeed \
        pytorch-lightning \
        wandb \
        tensorboard
}

install_utilities() {
    echo "Installing utilities..."
    source $VENV_DIR/bin/activate

    pip install \
        fastapi \
        uvicorn \
        websockets \
        aiohttp \
        requests \
        pyyaml \
        python-dotenv \
        rich \
        click \
        tqdm
}

download_base_models() {
    echo "Downloading base models..."
    source $VENV_DIR/bin/activate

    cd $MODELS_DIR/speech
    echo "Downloading Whisper models..."
    python3 << 'PYEOF'
import whisper
whisper.load_model("base")
whisper.load_model("medium")
PYEOF

    echo "Base models ready. Language-specific models will be downloaded on first use."
}

create_config() {
    echo "Creating configuration files..."

    cat > $INSTALL_DIR/configs/system.yaml << 'EOF'
system:
  install_dir: ~/llm_system
  models_dir: ~/llm_system/models
  cache_dir: ~/llm_system/cache
  log_dir: ~/llm_system/logs

gpu:
  primary:
    device: cuda:0
    model: RTX_3090
    memory: 24576
  remote:
    enabled: false
    hosts: []

llm:
  engine: vllm
  default_model: null
  max_tokens: 8192
  temperature: 0.7
  gpu_memory_utilization: 0.9

speech:
  input:
    engine: faster-whisper
    model: medium
    languages: [en, sr, bs, hr]
  output:
    engine: coqui-tts
    languages: [en, sr, bs, hr]

languages:
  supported: [en, sr, bs, hr]
  default: en
  auto_detect: true

agents:
  max_concurrent: 5
  memory_enabled: true
  tools_enabled: true

coding:
  assistant_enabled: true
  auto_complete: true
  context_window: 16000
EOF
}

create_launcher() {
    echo "Creating launcher script..."

    cat > $INSTALL_DIR/launch.sh << 'EOF'
#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $SCRIPT_DIR/venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=$SCRIPT_DIR/cache
export HF_HOME=$SCRIPT_DIR/cache

cd $SCRIPT_DIR
python3 -m system.main "$@"
EOF

    chmod +x $INSTALL_DIR/launch.sh
}

main() {
    create_directories
    install_system_deps
    setup_python_env
    install_llm_engines
    install_speech_engines
    install_multilang_support
    install_agent_framework
    install_dev_tools
    install_distributed_training
    install_utilities
    download_base_models
    create_config
    create_launcher

    echo ""
    echo "========================================"
    echo "LLM System Installation Complete!"
    echo "========================================"
    echo ""
    echo "Installation directory: $INSTALL_DIR"
    echo ""
    echo "To activate the environment:"
    echo "  source $VENV_DIR/bin/activate"
    echo ""
    echo "Next steps:"
    echo "  1. Run the system setup to download models"
    echo "  2. Configure remote GPUs if needed"
    echo "  3. Start the system"
    echo ""
}

main
