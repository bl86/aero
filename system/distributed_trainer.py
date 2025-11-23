import os
import ray
import torch
import torch.distributed as dist
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import json
from datetime import datetime

class DistributedTrainer:
    def __init__(self, config):
        self.config = config
        self.base_dir = Path(config['system']['install_dir']).expanduser()
        self.training_dir = self.base_dir / "training"
        self.training_dir.mkdir(exist_ok=True)

        self.primary_gpu = config['gpu']['primary']
        self.remote_hosts = config['gpu']['remote'].get('hosts', [])

        self.ray_initialized = False
        self.training_active = False

    def init_ray(self):
        if self.ray_initialized:
            return

        if len(self.remote_hosts) > 0:
            ray.init(address='auto')
        else:
            ray.init()

        self.ray_initialized = True

    def shutdown_ray(self):
        if self.ray_initialized:
            ray.shutdown()
            self.ray_initialized = False

    def setup_remote_nodes(self, hosts: List[str]):
        self.config['gpu']['remote']['hosts'] = hosts

        for host in hosts:
            print(f"Setting up remote node: {host}")
            self._setup_single_node(host)

    def _setup_single_node(self, host: str):
        setup_script = """
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
pip install ray torch transformers
ray start --address='auto' --num-gpus=4
"""

        print(f"Connecting to {host}...")

    def start_training(self, model_name: str = None, dataset_path: str = None,
                      config: Dict = None):

        if not model_name:
            print("Error: model_name required")
            return

        self.init_ray()

        training_config = config or self._default_training_config()

        @ray.remote(num_gpus=1)
        class TrainingWorker:
            def __init__(self, rank, world_size, model_name, training_config):
                self.rank = rank
                self.world_size = world_size
                self.model_name = model_name
                self.config = training_config

            def setup(self):
                torch.cuda.set_device(self.rank)

            def train_epoch(self, epoch):
                print(f"Worker {self.rank} training epoch {epoch}")

                return {
                    'epoch': epoch,
                    'rank': self.rank,
                    'loss': 0.5 - (epoch * 0.01)
                }

            def get_status(self):
                return {
                    'rank': self.rank,
                    'world_size': self.world_size,
                    'model': self.model_name
                }

        num_gpus = self._get_total_gpus()
        print(f"Starting distributed training on {num_gpus} GPUs")

        workers = [
            TrainingWorker.remote(i, num_gpus, model_name, training_config)
            for i in range(num_gpus)
        ]

        ray.get([w.setup.remote() for w in workers])

        epochs = training_config.get('epochs', 10)

        for epoch in range(epochs):
            results = ray.get([w.train_epoch.remote(epoch) for w in workers])

            avg_loss = sum(r['loss'] for r in results) / len(results)
            print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

            self._save_checkpoint(epoch, avg_loss)

        self.training_active = True

        return "Training started"

    def _default_training_config(self) -> Dict:
        return {
            'epochs': 10,
            'batch_size': 8,
            'learning_rate': 2e-5,
            'warmup_steps': 100,
            'gradient_accumulation_steps': 4,
            'fp16': True,
            'deepspeed': {
                'zero_optimization': {
                    'stage': 2
                }
            }
        }

    def _get_total_gpus(self) -> int:
        local_gpus = 1

        remote_gpus = len(self.remote_hosts) * 4

        return local_gpus + remote_gpus

    def _save_checkpoint(self, epoch: int, loss: float):
        checkpoint_dir = self.training_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.json"

        checkpoint_data = {
            'epoch': epoch,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def stop_training(self):
        if self.training_active:
            print("Stopping training...")
            self.training_active = False
            return "Training stopped"
        else:
            return "No active training"

    def get_status(self) -> str:
        status = {
            'ray_initialized': self.ray_initialized,
            'training_active': self.training_active,
            'primary_gpu': self.primary_gpu,
            'remote_hosts': len(self.remote_hosts),
            'total_gpus': self._get_total_gpus()
        }

        return json.dumps(status, indent=2)

    def configure_deepspeed(self, config: Dict = None):
        ds_config = config or {
            "train_batch_size": 32,
            "gradient_accumulation_steps": 4,
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu"
                },
                "offload_param": {
                    "device": "cpu"
                }
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 2e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 2e-5,
                    "warmup_num_steps": 100
                }
            }
        }

        ds_config_path = self.training_dir / "deepspeed_config.json"

        with open(ds_config_path, 'w') as f:
            json.dump(ds_config, f, indent=2)

        return str(ds_config_path)

    def train_lora(self, base_model: str, dataset_path: str,
                   lora_config: Dict = None):

        default_lora_config = {
            'r': 8,
            'lora_alpha': 32,
            'target_modules': ['q_proj', 'v_proj'],
            'lora_dropout': 0.1,
            'bias': 'none',
            'task_type': 'CAUSAL_LM'
        }

        config = lora_config or default_lora_config

        training_script = f"""
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

model = AutoModelForCausalLM.from_pretrained('{base_model}', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained('{base_model}')

lora_config = LoraConfig(
    r={config['r']},
    lora_alpha={config['lora_alpha']},
    target_modules={config['target_modules']},
    lora_dropout={config['lora_dropout']},
    bias='{config['bias']}',
    task_type='{config['task_type']}'
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()
"""

        script_path = self.training_dir / "train_lora.py"
        with open(script_path, 'w') as f:
            f.write(training_script)

        return str(script_path)

    def monitor_gpus(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )

            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(',')
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1].strip(),
                        'utilization': int(parts[2]),
                        'memory_used': int(parts[3]),
                        'memory_total': int(parts[4])
                    })

            return gpus

        except Exception as e:
            return [{'error': str(e)}]

    def get_training_logs(self) -> List[str]:
        log_dir = self.training_dir / "logs"

        if not log_dir.exists():
            return []

        logs = []
        for log_file in sorted(log_dir.glob("*.log")):
            with open(log_file, 'r') as f:
                logs.append(f.read())

        return logs

    def create_dataset_config(self, dataset_type: str = 'instruction'):
        configs = {
            'instruction': {
                'format': 'instruction',
                'columns': {
                    'input': 'instruction',
                    'output': 'response'
                },
                'prompt_template': 'Below is an instruction. Write a response.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}'
            },
            'conversation': {
                'format': 'conversation',
                'columns': {
                    'messages': 'messages'
                }
            },
            'completion': {
                'format': 'completion',
                'columns': {
                    'text': 'text'
                }
            }
        }

        config = configs.get(dataset_type, configs['instruction'])

        config_path = self.training_dir / f"dataset_config_{dataset_type}.json"

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        return str(config_path)
