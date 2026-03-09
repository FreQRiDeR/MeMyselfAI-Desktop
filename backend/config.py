"""
config.py
Configuration management for MeMyselfAI
"""

import json
import sys
from pathlib import Path
from typing import Optional


class Config:
    """Application configuration"""
    
    # Detect if running from PyInstaller bundle
    _is_bundled = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
    
    DEFAULT_CONFIG = {
        "backend_type": "local",  # local, ollama, or huggingface
        "llama_cpp_path": "bundled" if _is_bundled else "",
        "ollama_path": "bundled" if _is_bundled else "",  # Add this line
        "ollama_url": "http://localhost:11434",
        "hf_api_key": "",
        "default_model": "",
        "max_tokens": 512,
        "temperature": 0.7,
        "context_size": 2048,
        "threads": 4,
        # llama.cpp local backend tuning
        "llama_gpu_layers": "auto",      # -ngl (auto|all|N)
        "llama_batch_size": 2048,         # -b
        "llama_ubatch_size": 512,         # -ub
        "llama_threads_batch": 0,         # -tb (0 => follow --threads)
        "llama_flash_attn": "auto",      # -fa (auto|on|off)
        "llama_kv_offload": True,         # -kvo / -nkvo
        "llama_mmap": True,               # --mmap / --no-mmap
        "llama_mlock": False,             # --mlock
        "llama_numa": "disabled",        # --numa distribute|isolate|numactl
        "llama_priority": 0,              # --prio (-1..3)
        "llama_poll": 50,                 # --poll (0..100)
        "llama_extra_args": "",          # appended raw flags
        "save_conversations": True,
        "theme": "system",  # system, light, dark
        "font_family": "SF Pro",  # Chat font family
        "font_size": 13,  # Chat font size
        "inference_timeout": 300  # Seconds to wait for model response (all backends)
    }
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize configuration
        
        Args:
            config_file: Path to config JSON file
        """
        # Use Application Support for bundled app
        if getattr(sys, 'frozen', False):
            if sys.platform == 'darwin':
                config_dir = Path.home() / 'Library' / 'Application Support' / 'MeMyselfAI'
            else:
                config_dir = Path.home() / '.memyselfai'
            config_dir.mkdir(parents=True, exist_ok=True)
            self.config_file = config_dir / config_file
        else:
            self.config_file = Path(config_file)
        
        self.config = self.DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self):
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                    self.config.update(loaded)
                print(f"✅ [Config] Loaded from {self.config_file}")
            except Exception as e:
                print(f"⚠️  [Config] Failed to load: {e}")
                print(f"   Using defaults")
        else:
            print(f"ℹ️  [Config] No config file found, using defaults")
            self.save()  # Create default config file
    
    def save(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ [Config] Saved to {self.config_file}")
        except Exception as e:
            print(f"❌ [Config] Failed to save: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value and save"""
        self.config[key] = value
        self.save()
    
    def get_llama_cpp_path(self) -> Optional[str]:
        """Get llama.cpp binary path"""
        path = self.config.get("llama_cpp_path")
        # Allow 'bundled' as a valid value
        if path == 'bundled':
            return 'bundled'
        if path and Path(path).exists():
            return path
        return None
    
    def get_ollama_path(self) -> Optional[str]:
        """Get Ollama binary path"""
        path = self.config.get("ollama_path")
        # Allow 'bundled' as a valid value
        if path == 'bundled':
            return 'bundled'
        if path and Path(path).exists():
            return path
        return None
    
    def is_configured(self) -> bool:
        """Check if app is properly configured"""
        # For local backend, need llama.cpp path; for Ollama, need Ollama path
        backend_type = self.config.get("backend_type", "local")
        if backend_type == "local":
            return self.get_llama_cpp_path() is not None
        elif backend_type == "ollama":
            return self.get_ollama_path() is not None
        else:  # huggingface
            return True  # Only need API key


if __name__ == "__main__":
    # Test config
    config = Config("test_config.json")
    print("Config:", config.config)
    
    config.set("llama_cpp_path", "backend/bin/llama-server")
    print("Updated config:", config.config)
    
    # Cleanup
    Path("test_config.json").unlink(missing_ok=True)
