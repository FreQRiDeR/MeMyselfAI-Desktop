"""
llama_wrapper.py
Wraps the existing llama.cpp binary for inference using llama-server
"""

import sys
import subprocess
import json
import re
import shlex
import time
import requests
import threading
from pathlib import Path
from typing import Optional, Callable, Generator


class LlamaWrapper:
    """Wrapper around llama.cpp server"""
    
    # Class variable to track instances (for debugging)
    _instance_count = 0
    
    def __init__(self, llama_cpp_path: str, tuning: Optional[dict] = None):
        """
        Initialize the wrapper
        
        Args:
            llama_cpp_path: Path to llama-server binary (or 'bundled' to use bundled version)
            tuning: Optional local backend tuning settings from config
        """
        tuning = tuning or {}

        # Track instance creation
        LlamaWrapper._instance_count += 1
        self.instance_id = LlamaWrapper._instance_count
        print(f"🔧 [LlamaWrapper #{self.instance_id}] Creating new instance")

        def _cfg_int(key: str, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
            try:
                value = int(tuning.get(key, default))
            except (TypeError, ValueError):
                value = default
            if min_value is not None:
                value = max(min_value, value)
            if max_value is not None:
                value = min(max_value, value)
            return value

        self.context_size = _cfg_int("context_size", 2048, min_value=0)
        self.threads = _cfg_int("threads", 4, min_value=1)
        self.batch_size = _cfg_int("llama_batch_size", 2048, min_value=1)
        self.ubatch_size = _cfg_int("llama_ubatch_size", 512, min_value=1)
        self.threads_batch = _cfg_int("llama_threads_batch", 0, min_value=0)
        self.flash_attn = str(tuning.get("llama_flash_attn", "auto")).lower()
        self.kv_offload = bool(tuning.get("llama_kv_offload", True))
        self.use_mmap = bool(tuning.get("llama_mmap", True))
        self.use_mlock = bool(tuning.get("llama_mlock", False))
        self.numa_mode = str(tuning.get("llama_numa", "disabled")).lower()
        self.priority = _cfg_int("llama_priority", 0, min_value=-1, max_value=3)
        self.poll = _cfg_int("llama_poll", 50, min_value=0, max_value=100)
        self.extra_args = str(tuning.get("llama_extra_args", "")).strip()
        self.request_timeout = _cfg_int("inference_timeout", 300, min_value=30, max_value=3600)

        if self.ubatch_size > self.batch_size:
            self.ubatch_size = self.batch_size

        if self.flash_attn not in {"auto", "on", "off"}:
            self.flash_attn = "auto"
        if self.numa_mode not in {"disabled", "distribute", "isolate", "numactl"}:
            self.numa_mode = "disabled"

        raw_gpu_layers = str(tuning.get("llama_gpu_layers", "auto")).strip().lower()
        if raw_gpu_layers in {"auto", "all"}:
            self.gpu_layers = raw_gpu_layers
        else:
            try:
                self.gpu_layers = str(int(raw_gpu_layers))
            except ValueError:
                self.gpu_layers = "auto"
        
        # Handle the case where user selected llama-cli instead of llama-server
        if 'llama-cli' in llama_cpp_path or 'llama-simple-chat' in llama_cpp_path:
            print(f"⚠️  [LlamaWrapper #{self.instance_id}] Detected llama-cli, converting to llama-server")
            # Try to find llama-server in the same directory
            cli_path = Path(llama_cpp_path)
            server_path = cli_path.parent / 'llama-server'
            if server_path.exists():
                llama_cpp_path = str(server_path)
                print(f"✅ [LlamaWrapper #{self.instance_id}] Using llama-server: {llama_cpp_path}")
            else:
                # Fallback: try to find any llama-server
                possible_server_paths = [
                    cli_path.parent / 'llama-server',
                    Path('backend/bin/llama-server'),
                    Path('./llama-server'),
                ]
                for path in possible_server_paths:
                    if path.exists():
                        llama_cpp_path = str(path)
                        print(f"✅ [LlamaWrapper #{self.instance_id}] Found llama-server: {llama_cpp_path}")
                        break
        
        # Check if running from PyInstaller bundle and should use bundled version
        if llama_cpp_path == 'bundled' or (hasattr(sys, '_MEIPASS') and not llama_cpp_path):
            # Running from PyInstaller bundle - look for bundled llama-server
            if hasattr(sys, '_MEIPASS'):
                # Try multiple possible locations
                bundle_dir = sys._MEIPASS
                possible_paths = [
                    # Current spec bundles binaries under backend/bin in Frameworks/_internal.
                    Path(bundle_dir) / 'backend' / 'bin' / 'llama-server',
                    # Backward-compatible legacy locations.
                    Path(bundle_dir) / 'llama' / 'llama-server',
                    Path(bundle_dir) / '../Frameworks/llama' / 'llama-server',
                    Path(bundle_dir) / 'llama-server',
                    # Additional macOS bundle variants.
                    Path(bundle_dir) / '../Frameworks' / 'backend' / 'bin' / 'llama-server',
                    Path(bundle_dir) / '../MacOS' / 'backend' / 'bin' / 'llama-server',
                ]
                
                bundled_path = None
                for path in possible_paths:
                    if path.exists():
                        bundled_path = path.resolve()
                        break
                
                if bundled_path:
                    self.llama_cpp_path = bundled_path
                    print(f"✅ Using bundled llama-server: {bundled_path}")
                else:
                    # If not found, show the primary expected location first.
                    expected = Path(bundle_dir) / 'backend' / 'bin' / 'llama-server'
                    raise FileNotFoundError(
                        f"Bundled llama-server not found!\n"
                        f"Expected at: {expected}\n"
                        f"Bundle dir: {bundle_dir}\n"
                        f"Also checked: {[str(p) for p in possible_paths]}"
                    )
            else:
                raise FileNotFoundError("Not running from bundle, cannot use 'bundled' path")
        else:
            self.llama_cpp_path = Path(llama_cpp_path)
        
        # Verify it's actually llama-server
        if 'llama-server' not in str(self.llama_cpp_path):
            raise ValueError(f"Expected llama-server binary, got: {self.llama_cpp_path}")
        
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port = 8080
        self.is_generating = False
        self.current_model: Optional[str] = None
        self.server_ready = False
        self.last_generation_stats = {}
        
        if not self.llama_cpp_path.exists():
            raise FileNotFoundError(f"llama-server not found at: {self.llama_cpp_path}")
    
    def check_model_file(self, model_path: str) -> bool:
        """Check if model file exists and is readable"""
        path = Path(model_path)
        return path.exists() and path.is_file() and path.suffix == '.gguf'

    def _build_server_command(self, model_path: str):
        """Build llama-server command from tuning settings."""
        cmd = [
            str(self.llama_cpp_path),
            '-m', model_path,
            '-c', str(self.context_size),
            '-t', str(self.threads),
            '-b', str(self.batch_size),
            '-ub', str(self.ubatch_size),
            '-ngl', self.gpu_layers,
            '-fa', self.flash_attn,
            '--prio', str(self.priority),
            '--poll', str(self.poll),
        ]

        if self.threads_batch > 0:
            cmd.extend(['-tb', str(self.threads_batch)])

        cmd.append('-kvo' if self.kv_offload else '-nkvo')
        cmd.append('--mmap' if self.use_mmap else '--no-mmap')
        if self.use_mlock:
            cmd.append('--mlock')
        if self.numa_mode in {"distribute", "isolate", "numactl"}:
            cmd.extend(['--numa', self.numa_mode])

        # Keep port/host arguments last and controlled by wrapper.
        if self.extra_args:
            try:
                cmd.extend(shlex.split(self.extra_args))
            except ValueError as exc:
                print(f"⚠️  [LlamaWrapper #{self.instance_id}] Ignoring invalid extra args: {exc}")

        cmd.extend([
            '--port', str(self.server_port),
            '--host', '127.0.0.1',
            '--no-webui',
        ])
        return cmd
    
    def _start_server(self, model_path: str):
        """Start the llama-server process"""
        # Check if we already have a server running with the same model
        if (self.server_process is not None and 
            self.server_process.poll() is None and  # Process still running
            self.current_model == model_path and   # Same model
            self.server_ready):  # Server is ready
            print(f"♻️  [LlamaWrapper #{self.instance_id}] Reusing existing server (PID: {self.server_process.pid})")
            return True
        
        # Clean up any existing server
        if self.server_process is not None:
            print(f"🧹 [LlamaWrapper #{self.instance_id}] Cleaning up old server")
            self._stop_server()
        
        # Find available port
        self.server_port = self._find_available_port()
        
        # Build command for llama-server
        cmd = self._build_server_command(model_path)
        
        print(f"🚀 [LlamaWrapper #{self.instance_id}] Starting server: {' '.join(cmd)}")
        
        # Start server process with Metal/GPU env vars
        import os
        gpu_env = {
            **os.environ,
            'LLAMA_METAL': '0',          # Enable Metal backend
            'GGML_METAL_FULL_OFFLOAD': '0',  # Offload all layers to GPU
        }
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=gpu_env
            )
        except Exception as e:
            print(f"❌ [LlamaWrapper #{self.instance_id}] Failed to start server: {e}")
            self.server_process = None
            return False
        
        self.current_model = model_path
        self.server_ready = False
        
        # Wait for server to be ready
        if self._wait_for_server_ready():
            print(f"✅ [LlamaWrapper #{self.instance_id}] Server started (PID: {self.server_process.pid}) on port {self.server_port}")
            return True
        else:
            print(f"❌ [LlamaWrapper #{self.instance_id}] Server failed to start")
            # Print stderr to see what went wrong
            try:
                stderr_output = self.server_process.stderr.read()
                if stderr_output:
                    print(f"stderr: {stderr_output}")
            except:
                pass
            self._stop_server()
            return False
    
    def _find_available_port(self) -> int:
        """Find an available port for the server"""
        import socket
        for port in range(8080, 8100):  # Try ports 8080-8099 first
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('localhost', port))
                sock.close()
                return port
            except OSError:
                continue
        
        # If those are taken, find any available port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('localhost', 0))
        port = sock.getsockname()[1]
        sock.close()
        return port
    
    def _wait_for_server_ready(self, timeout: int = 30) -> bool:
        """Wait for server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f'http://127.0.0.1:{self.server_port}/health', timeout=1)
                if response.status_code == 200:
                    self.server_ready = True
                    return True
            except:
                pass
            time.sleep(0.5)
        return False
    
    def _stop_server(self):
        """Stop the server process"""
        if self.server_process:
            print(f"🛑 [LlamaWrapper #{self.instance_id}] Stopping server (PID: {self.server_process.pid})")
            try:
                # Try graceful shutdown first
                try:
                    requests.post(f'http://127.0.0.1:{self.server_port}/shutdown', timeout=2)
                    self.server_process.wait(timeout=2)
                except:
                    # Force terminate if graceful shutdown fails
                    self.server_process.terminate()
                    try:
                        self.server_process.wait(timeout=1)
                    except:
                        self.server_process.kill()
                        self.server_process.wait()
            except Exception as e:
                print(f"⚠️  [LlamaWrapper #{self.instance_id}] Error stopping server: {e}")
            finally:
                self.server_process = None
                self.current_model = None
                self.server_ready = False
    
    def generate_streaming(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        callback: Optional[Callable[[str], None]] = None,
        messages: list = None
    ) -> Generator[str, None, None]:
        """
        Generate response with streaming output using HTTP API.
        History is managed by the caller (main_window) and passed in via messages.

        Args:
            model_path: Path to .gguf model file
            prompt: Current user prompt (used as fallback if messages is None)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            callback: Optional callback for each token
            messages: Full conversation history from main_window. If None,
                      falls back to a bare single-turn user prompt.

        Yields:
            Generated tokens as they arrive
        """
        print(f"💬 [LlamaWrapper #{self.instance_id}] Generating response for: {prompt[:50]}...")

        if not self.check_model_file(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            if not self._start_server(model_path):
                raise Exception("Failed to start model server")

            self.is_generating = True
            request_start = time.time()
            first_token_time = None
            stream_usage = None

            # Use caller-supplied history; fall back to bare prompt if not provided
            chat_messages = messages if messages else [{"role": "user", "content": prompt}]

            url = f'http://127.0.0.1:{self.server_port}/v1/chat/completions'
            payload = {
                "messages": chat_messages,
                "stream": True,
                "max_tokens": max_tokens,
                "temperature": temperature,
                # Ask for token usage in the final streaming chunk when supported.
                "stream_options": {"include_usage": True},
            }

            response = requests.post(url, json=payload, stream=True, timeout=self.request_timeout)
            response.raise_for_status()

            full_response = ""

            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            if isinstance(chunk, dict) and "usage" in chunk:
                                stream_usage = chunk.get("usage") or stream_usage
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                    full_response += content
                                    if callback:
                                        callback(content)
                                    yield content
                        except json.JSONDecodeError:
                            continue

            request_end = time.time()
            prompt_seconds = None
            generation_seconds = None
            if first_token_time is not None:
                prompt_seconds = max(1e-9, first_token_time - request_start)
                generation_seconds = max(1e-9, request_end - first_token_time)
            else:
                prompt_seconds = max(1e-9, request_end - request_start)

            prompt_tokens = None
            completion_tokens = None
            if isinstance(stream_usage, dict):
                prompt_tokens = stream_usage.get("prompt_tokens")
                completion_tokens = stream_usage.get("completion_tokens")

            # Fallback estimates if usage is not returned by the server.
            if prompt_tokens is None:
                prompt_chars = sum(len(m.get("content", "")) for m in chat_messages)
                prompt_tokens = max(1, prompt_chars // 4)
            if completion_tokens is None:
                completion_tokens = max(1, len(full_response) // 4)

            prompt_tps = None
            generation_tps = None
            if prompt_seconds and prompt_tokens:
                prompt_tps = prompt_tokens / prompt_seconds
            if generation_seconds and completion_tokens:
                generation_tps = completion_tokens / generation_seconds

            self.last_generation_stats = {
                "prompt_tps": prompt_tps,
                "generation_tps": generation_tps,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            prompt_tps_text = f"{prompt_tps:.1f}" if prompt_tps is not None else "n/a"
            generation_tps_text = f"{generation_tps:.1f}" if generation_tps is not None else "n/a"

            print(
                f"✅ [LlamaWrapper #{self.instance_id}] Response complete ({len(full_response)} chars) "
                f"[prompt {prompt_tps_text} t/s | generation {generation_tps_text} t/s]"
            )

        except Exception as e:
            print(f"❌ [LlamaWrapper #{self.instance_id}] Exception during generation: {e}")
            raise
        finally:
            self.is_generating = False

    def get_last_generation_stats(self) -> dict:
        """Return stats from the most recent streamed generation."""
        return dict(self.last_generation_stats)
    
    def generate(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response (blocking, returns full response)
        
        Args:
            model_path: Path to .gguf model file
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Complete generated response
        """
        print(f"🔄 [LlamaWrapper #{self.instance_id}] Generate (blocking) for: {prompt[:50]}...")
        response_parts = []
        for token in self.generate_streaming(model_path, prompt, max_tokens, temperature):
            response_parts.append(token)
        result = ''.join(response_parts).strip()
        print(f"✅ [LlamaWrapper #{self.instance_id}] Generate complete ({len(result)} chars)")
        return result
    
    def stop_generation(self):
        """Stop current generation"""
        # For HTTP API, we can't really stop a request in progress
        # But we can mark that we're not interested in the response anymore
        self.is_generating = False
    
    def cleanup(self):
        """Clean up resources — called by UnifiedBackend on backend switch or app quit"""
        self.shutdown()

    def shutdown(self):
        """Properly shut down the wrapper"""
        print(f"🔌 [LlamaWrapper #{self.instance_id}] Shutting down...")
        self._stop_server()
    
    def __del__(self):
        """Cleanup on deletion"""
        print(f"🗑️  [LlamaWrapper #{self.instance_id}] Destructor called")
        self.shutdown()
