"""
unified_backend.py
Unified backend supporting Local llama.cpp, Ollama, and HuggingFace
"""

import sys
import json
import requests
import subprocess
import time
from enum import Enum
from typing import Generator, Optional, Callable
from pathlib import Path


class BackendType(Enum):
    """Available backend types"""
    LOCAL = "local"           # Local llama.cpp
    OLLAMA = "ollama"         # Ollama API (local or remote)
    HUGGINGFACE = "huggingface"  # HuggingFace Inference API


class UnifiedBackend:
    """Unified interface for multiple LLM backends"""
    
    def __init__(self, backend_type: BackendType, **config):
        """
        Initialize backend
        
        Args:
            backend_type: Type of backend to use
            **config: Backend-specific configuration
                For LOCAL: llama_cpp_path
                For OLLAMA: ollama_url (default: http://localhost:11434), ollama_path
                For HUGGINGFACE: api_key
        """
        self.backend_type = backend_type
        self.config = config
        self.inference_timeout = int(config.get('inference_timeout', 300))
        self.ollama_process = None  # To track Ollama process
        
        # Initialize backend-specific components
        if backend_type == BackendType.LOCAL:
            from backend.llama_wrapper import LlamaWrapper
            llama_path = config.get('llama_cpp_path', 'bundled')
            self.local_wrapper = LlamaWrapper(llama_path, tuning=config)
        elif backend_type == BackendType.OLLAMA:
            self.ollama_url = config.get('ollama_url', 'http://localhost:11434')
            self.ollama_path = config.get('ollama_path', 'bundled')
            self._start_ollama_if_needed()
        elif backend_type == BackendType.HUGGINGFACE:
            self.hf_api_key = config.get('api_key')
            if not self.hf_api_key:
                raise ValueError("HuggingFace API key required")
    
    def _start_ollama_if_needed(self):
        """Start Ollama serve process if using bundled Ollama"""
        if self.ollama_path == 'bundled' or (hasattr(sys, '_MEIPASS') and not self.ollama_path):
            # Don't spawn a new process if Ollama is already responding on the port
            if self.test_ollama_connection(self.ollama_url):
                print("ℹ️  Ollama already running, skipping start")
                return

            ollama_binary = self._find_bundled_ollama()
            if ollama_binary:
                try:
                    print(f"🚀 Starting bundled Ollama: {ollama_binary}")
                    self.ollama_process = subprocess.Popen(
                        [str(ollama_binary), 'serve'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                    # Wait up to 10s for Ollama to be ready instead of blind sleep
                    for _ in range(20):
                        time.sleep(0.5)
                        if self.test_ollama_connection(self.ollama_url):
                            print("✅ Ollama started successfully")
                            return
                    print("⚠️  Ollama started but not yet responding")
                except Exception as e:
                    print(f"⚠️  Failed to start bundled Ollama: {e}")
    
    def _find_bundled_ollama(self):
        """Find bundled Ollama binary"""
        if hasattr(sys, '_MEIPASS'):
            # Running from PyInstaller bundle
            bundle_dir = sys._MEIPASS
            possible_paths = [
                Path(bundle_dir) / 'backend' / 'bin' / 'ollama',
                Path(bundle_dir) / 'ollama',
            ]
        else:
            # Development mode
            possible_paths = [
                Path(__file__).parent / 'bin' / 'ollama',
                Path('./backend/bin/ollama'),
                Path('./ollama'),
            ]
        
        for path in possible_paths:
            if path.exists():
                return path
        return None
    
    def generate_streaming(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        callback: Optional[Callable[[str], None]] = None,
        messages: list = None
    ) -> Generator[str, None, None]:
        """
        Generate response with streaming.

        Args:
            model: Model name/path
            prompt: Current user prompt (used by LOCAL and HF backends)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            callback: Optional callback for each token
            messages: Full conversation history as [{"role": ..., "content": ...}].
                      When provided and using Ollama, sent to /api/chat for context.

        Yields:
            Generated tokens as they arrive
        """
        if self.backend_type == BackendType.LOCAL:
            yield from self._local_generate(model, prompt, max_tokens, temperature, callback, messages)
        elif self.backend_type == BackendType.OLLAMA:
            yield from self._ollama_generate(model, prompt, max_tokens, temperature, callback, messages)
        elif self.backend_type == BackendType.HUGGINGFACE:
            yield from self._hf_generate(model, prompt, max_tokens, temperature, callback)
    
    def _local_generate(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        callback: Optional[Callable[[str], None]],
        messages: list = None
    ) -> Generator[str, None, None]:
        """Generate using local llama.cpp"""
        yield from self.local_wrapper.generate_streaming(
            model_path, prompt, max_tokens, temperature, callback, messages
        )
    
    def _ollama_generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        callback: Optional[Callable[[str], None]],
        messages: list = None
    ) -> Generator[str, None, None]:
        """Generate using Ollama /api/chat with full conversation history"""
        try:
            # Use provided history, or wrap the bare prompt as a single user turn
            chat_messages = messages if messages else [{"role": "user", "content": prompt}]

            response = requests.post(
                f'{self.ollama_url}/api/chat',
                json={
                    "model": model,
                    "messages": chat_messages,
                    "stream": True,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                },
                stream=True,
                timeout=self.inference_timeout
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if not data.get('done', False):
                            # /api/chat returns tokens under message.content
                            token = data.get('message', {}).get('content', '')
                            if token:
                                if callback:
                                    callback(token)
                                yield token
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")
    
    def _hf_generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        callback: Optional[Callable[[str], None]]
    ) -> Generator[str, None, None]:
        """Generate using HuggingFace Inference API"""
        try:
            response = requests.post(
                f'https://api-inference.huggingface.co/models/{model}',
                headers={
                    "Authorization": f"Bearer {self.hf_api_key}"
                },
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "return_full_text": False
                    },
                    "options": {
                        "use_cache": False
                    }
                },
                stream=True,
                timeout=self.inference_timeout
            )
            response.raise_for_status()
            
            # HuggingFace can return either SSE or plain JSON
            # Try SSE first
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    
                    # SSE format: "data: {...}"
                    if line_str.startswith('data: '):
                        try:
                            data = json.loads(line_str[6:])
                            if isinstance(data, dict) and 'token' in data:
                                token = data['token'].get('text', '')
                            elif isinstance(data, dict) and 'generated_text' in data:
                                # Non-streaming response
                                text = data['generated_text']
                                if callback:
                                    callback(text)
                                yield text
                                return
                            else:
                                continue
                            
                            if token:
                                if callback:
                                    callback(token)
                                yield token
                        except json.JSONDecodeError:
                            continue
                    # Plain JSON (non-streaming fallback)
                    else:
                        try:
                            data = json.loads(line_str)
                            if isinstance(data, list) and len(data) > 0:
                                text = data[0].get('generated_text', '')
                            elif isinstance(data, dict):
                                text = data.get('generated_text', '')
                            else:
                                continue
                            
                            if text:
                                if callback:
                                    callback(text)
                                yield text
                                return
                        except json.JSONDecodeError:
                            continue
                            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HuggingFace API error: {e}")

    def get_last_generation_stats(self) -> dict:
        """Return stats from the last generation if backend provides them."""
        if self.backend_type == BackendType.LOCAL and hasattr(self, "local_wrapper"):
            if hasattr(self.local_wrapper, "get_last_generation_stats"):
                return self.local_wrapper.get_last_generation_stats()
        return {}
    
    def stop_generation(self):
        """Stop current generation"""
        if self.backend_type == BackendType.LOCAL:
            self.local_wrapper.stop_generation()
        # Ollama and HF don't need explicit stopping (HTTP request ends)
    
    def cleanup(self):
        """Clean up backend resources"""
        # Clean up LOCAL llama-server process via the wrapper
        if self.backend_type == BackendType.LOCAL and hasattr(self, 'local_wrapper'):
            try:
                self.local_wrapper.cleanup()
            except Exception as e:
                print(f"⚠️  llama_wrapper cleanup error: {e}")

        # Clean up bundled Ollama process
        if self.ollama_process:
            if self.ollama_process.poll() is None:  # still running
                print("🛑 Stopping Ollama process...")
                self.ollama_process.terminate()
                try:
                    self.ollama_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print("⚠️  Ollama didn't terminate cleanly, killing...")
                    self.ollama_process.kill()
                    self.ollama_process.wait()
                print("✅ Ollama stopped")
            self.ollama_process = None
    
    @staticmethod
    def get_ollama_models(ollama_url: str = 'http://localhost:11434') -> list:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f'{ollama_url}/api/tags', timeout=5)
            response.raise_for_status()
            data = response.json()
            return [m['name'] for m in data.get('models', [])]
        except Exception as e:
            print(f"⚠️  Failed to fetch Ollama models: {e}")
            return []
    
    @staticmethod
    def test_ollama_connection(ollama_url: str = 'http://localhost:11434') -> bool:
        """Test if Ollama is running and accessible"""
        try:
            response = requests.get(f'{ollama_url}/api/tags', timeout=2)
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def test_hf_api_key(api_key: str) -> bool:
        """Test if HuggingFace API key is valid"""
        try:
            response = requests.get(
                'https://huggingface.co/api/whoami',
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


if __name__ == "__main__":
    # Test backends
    print("Testing UnifiedBackend...")
    
    # Test Ollama
    if UnifiedBackend.test_ollama_connection():
        print("\n✅ Ollama is running!")
        models = UnifiedBackend.get_ollama_models()
        print(f"Available models: {models}")
        
        if models:
            backend = UnifiedBackend(BackendType.OLLAMA)
            print(f"\nTesting with {models[0]}...")
            response = ""
            for token in backend.generate_streaming(models[0], "Say hello!", max_tokens=50):
                response += token
                print(token, end='', flush=True)
            print(f"\n\nComplete response: {response}")
    else:
        print("\n❌ Ollama not running (install from https://ollama.ai)")
    
    # Test HuggingFace (need API key)
    print("\n" + "="*60)
    print("HuggingFace test requires API key (get from https://huggingface.co/settings/tokens)")
    print("Set HF_API_KEY environment variable to test")
    
    import os
    hf_key = os.environ.get('HF_API_KEY')
    if hf_key:
        if UnifiedBackend.test_hf_api_key(hf_key):
            print("✅ HuggingFace API key is valid!")
        else:
            print("❌ Invalid HuggingFace API key")
    else:
        print("ℹ️  Skipping HuggingFace test (no API key)")
