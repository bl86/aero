#!/bin/bash

set -e

echo "=========================================="
echo "Local LLM System - Quick Start"
echo "=========================================="
echo ""

INSTALL_DIR="$HOME/llm_system"

if [ ! -d "$INSTALL_DIR" ]; then
    echo "System not installed. Running installation..."
    echo ""

    read -p "Do you want to install GPU drivers? (y/n): " install_gpu
    if [ "$install_gpu" = "y" ]; then
        echo "Installing GPU support..."
        chmod +x setup_gpu.sh
        ./setup_gpu.sh

        echo ""
        echo "GPU setup complete. Please reboot your system:"
        echo "  sudo reboot"
        echo ""
        echo "After reboot, run this script again to continue."
        exit 0
    fi

    echo "Installing LLM system..."
    chmod +x install_llm.sh
    ./install_llm.sh

    echo ""
    echo "Installation complete!"
    echo ""
fi

source $INSTALL_DIR/venv/bin/activate

if [ ! -d "$INSTALL_DIR/models/llm" ] || [ -z "$(ls -A $INSTALL_DIR/models/llm)" ]; then
    echo "No models found. Would you like to download models?"
    echo ""
    echo "Recommended options:"
    echo "  1. Small model for testing (TinyLlama 1.1B) - ~2GB"
    echo "  2. Coding model (DeepSeek Coder 6.7B) - ~13GB"
    echo "  3. General purpose model (Mistral 7B) - ~14GB"
    echo "  4. Skip and download later"
    echo ""

    read -p "Select option (1-4): " model_choice

    case $model_choice in
        1)
            echo "Downloading TinyLlama..."
            python3 download_models.py << EOF
2
0
EOF
            ;;
        2)
            echo "Downloading DeepSeek Coder..."
            python3 download_models.py << EOF
1
0
EOF
            ;;
        3)
            echo "Downloading Mistral 7B..."
            python3 download_models.py << EOF
5
0
EOF
            ;;
        *)
            echo "Skipping model download"
            ;;
    esac
fi

echo ""
echo "=========================================="
echo "System Ready!"
echo "=========================================="
echo ""
echo "Available modes:"
echo "  1. Interactive mode (chat with LLM)"
echo "  2. API server mode"
echo "  3. Download more models"
echo "  4. Configure remote GPUs"
echo ""

read -p "Select mode (1-4): " mode_choice

case $mode_choice in
    1)
        echo ""
        echo "Starting interactive mode..."
        echo ""
        python3 -m system.main --mode interactive
        ;;
    2)
        read -p "Port (default 8000): " port
        port=${port:-8000}

        echo ""
        echo "Starting API server on port $port..."
        echo "API documentation: http://localhost:$port/docs"
        echo ""
        python3 -m system.main --mode server --port $port
        ;;
    3)
        echo ""
        python3 download_models.py
        ;;
    4)
        echo ""
        echo "Remote GPU Configuration"
        echo "========================"
        echo ""
        echo "Enter remote hosts (one per line, empty line to finish):"

        hosts=()
        while true; do
            read -p "Host: " host
            if [ -z "$host" ]; then
                break
            fi
            hosts+=("$host")
        done

        if [ ${#hosts[@]} -gt 0 ]; then
            echo ""
            echo "Configuring ${#hosts[@]} remote hosts..."

            hosts_json="["
            for i in "${!hosts[@]}"; do
                hosts_json+="\"${hosts[$i]}\""
                if [ $i -lt $((${#hosts[@]} - 1)) ]; then
                    hosts_json+=","
                fi
            done
            hosts_json+="]"

            python3 << EOF
import yaml
from pathlib import Path

config_path = Path.home() / "llm_system" / "configs" / "system.yaml"

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config['gpu']['remote']['enabled'] = True
config['gpu']['remote']['hosts'] = ${hosts_json}

with open(config_path, 'w') as f:
    yaml.dump(config, f)

print("Remote GPU configuration saved!")
EOF
        fi
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac
