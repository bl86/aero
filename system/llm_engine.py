import torch
import os
from pathlib import Path
from typing import Optional, List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

class LLMEngine:
    def __init__(self, config):
        self.config = config
        self.base_dir = Path(config['system']['install_dir']).expanduser()
        self.models_dir = self.base_dir / "models" / "llm"
        self.cache_dir = self.base_dir / "cache"

        os.environ['TRANSFORMERS_CACHE'] = str(self.cache_dir)
        os.environ['HF_HOME'] = str(self.cache_dir)

        self.device = config['gpu']['primary']['device']
        self.model = None
        self.tokenizer = None
        self.vllm_engine = None

        self.loaded_model_name = None

    def load_model(self, model_name: str, use_vllm: bool = True):
        if self.loaded_model_name == model_name:
            return

        self.unload_model()

        model_path = self.models_dir / model_name

        if not model_path.exists():
            model_path = model_name

        if use_vllm and torch.cuda.is_available():
            self._load_vllm_model(str(model_path))
        else:
            self._load_hf_model(str(model_path))

        self.loaded_model_name = model_name

    def _load_vllm_model(self, model_path: str):
        gpu_memory = self.config['llm'].get('gpu_memory_utilization', 0.9)

        self.vllm_engine = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory,
            trust_remote_code=True,
            dtype='auto',
            max_model_len=self.config['llm'].get('max_tokens', 8192)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _load_hf_model(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map='auto',
            trust_remote_code=True
        )

    def generate(self, prompt: str, max_tokens: int = None,
                temperature: float = None, stop: List[str] = None) -> str:

        if not self.model and not self.vllm_engine:
            return "No model loaded. Please load a model first."

        max_tokens = max_tokens or self.config['llm'].get('max_tokens', 2048)
        temperature = temperature or self.config['llm'].get('temperature', 0.7)

        if self.vllm_engine:
            return self._generate_vllm(prompt, max_tokens, temperature, stop)
        else:
            return self._generate_hf(prompt, max_tokens, temperature, stop)

    def _generate_vllm(self, prompt: str, max_tokens: int,
                      temperature: float, stop: List[str]) -> str:

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop or []
        )

        outputs = self.vllm_engine.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def _generate_hf(self, prompt: str, max_tokens: int,
                    temperature: float, stop: List[str]) -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if response.startswith(prompt):
            response = response[len(prompt):]

        return response.strip()

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = None,
            temperature: float = None) -> str:

        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = self._format_chat_manual(messages)

        return self.generate(prompt, max_tokens, temperature)

    def _format_chat_manual(self, messages: List[Dict[str, str]]) -> str:
        formatted = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted.append(f"System: {content}")
            elif role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")

        formatted.append("Assistant:")
        return "\n\n".join(formatted)

    def generate_code(self, description: str, language: str = "python") -> str:
        prompt = f"""Generate {language} code for the following task:

Task: {description}

Provide clean, well-documented code with comments.

Code:
"""
        return self.generate(prompt, max_tokens=4096, temperature=0.2)

    def unload_model(self):
        if self.vllm_engine:
            del self.vllm_engine
            self.vllm_engine = None

        if self.model:
            del self.model
            self.model = None

        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()

        self.loaded_model_name = None

    def get_available_models(self) -> List[str]:
        if not self.models_dir.exists():
            return []

        models = []
        for item in self.models_dir.iterdir():
            if item.is_dir():
                models.append(item.name)

        return models

    def download_model(self, model_name: str):
        from huggingface_hub import snapshot_download

        local_path = self.models_dir / model_name.replace('/', '_')

        if local_path.exists():
            return str(local_path)

        snapshot_download(
            repo_id=model_name,
            local_dir=str(local_path),
            local_dir_use_symlinks=False
        )

        return str(local_path)

    def get_model_info(self) -> Dict:
        return {
            'loaded_model': self.loaded_model_name,
            'engine': 'vllm' if self.vllm_engine else 'transformers',
            'device': self.device,
            'available_models': self.get_available_models()
        }
