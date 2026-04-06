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
import os
import signal
from pathlib import Path
from typing import Optional, Callable, Generator
from backend.process_utils import background_process_kwargs


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
                    Path('backend/bin/linux/llama-server'),
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
                possible_paths = []

                if sys.platform == 'win32':
                    possible_paths.extend([
                        Path(bundle_dir) / 'backend' / 'bin' / 'windows' / 'llama-server.exe',
                        Path(bundle_dir) / 'backend' / 'bin' / 'llama-server.exe',
                        Path(bundle_dir) / 'llama-server.exe',
                    ])

                possible_paths.extend([
                    # Current Unix-like bundle layouts.
                    Path(bundle_dir) / 'backend' / 'bin' / 'llama-server',
                    Path(bundle_dir) / 'backend' / 'bin' / 'linux' / 'llama-server',
                    # Backward-compatible legacy locations.
                    Path(bundle_dir) / 'llama' / 'llama-server',
                    Path(bundle_dir) / '../Frameworks/llama' / 'llama-server',
                    Path(bundle_dir) / 'llama-server',
                    # Additional macOS bundle variants.
                    Path(bundle_dir) / '../Frameworks' / 'backend' / 'bin' / 'llama-server',
                    Path(bundle_dir) / '../MacOS' / 'backend' / 'bin' / 'llama-server',
                ])
                
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
                    expected = (
                        Path(bundle_dir) / 'backend' / 'bin' / 'windows' / 'llama-server.exe'
                        if sys.platform == 'win32'
                        else Path(bundle_dir) / 'backend' / 'bin' / 'llama-server'
                    )
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
        self.stop_requested = False
        self._active_response = None
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

    def _build_server_env(self):
        """Build environment for launching llama-server with bundled shared libraries."""
        env = dict(os.environ)
        binary_dir = str(self.llama_cpp_path.parent.resolve())

        if sys.platform.startswith("linux"):
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                f"{binary_dir}{os.pathsep}{existing}" if existing else binary_dir
            )
        elif sys.platform == "darwin":
            existing = env.get("DYLD_LIBRARY_PATH", "")
            env["DYLD_LIBRARY_PATH"] = (
                f"{binary_dir}{os.pathsep}{existing}" if existing else binary_dir
            )
            # Only relevant on macOS builds with Metal support.
            env.setdefault("GGML_METAL_FULL_OFFLOAD", "0")

        return env
    
    def _start_server(self, model_path: str):
        """Start the llama-server process"""
        self.stop_requested = False
        # Check if we already have a server running with the same model
        if (self.server_process is not None and
            self.server_process.poll() is None and  # Process still running
            self.current_model == model_path):      # Same model
            # If server_ready was cleared, re-check health before tearing down.
            if not self.server_ready:
                try:
                    response = requests.get(f'http://127.0.0.1:{self.server_port}/health', timeout=1)
                    if response.status_code == 200:
                        self.server_ready = True
                except Exception:
                    pass
            if self.server_ready:
                print(f"♻️  [LlamaWrapper #{self.instance_id}] Reusing existing server (PID: {self.server_process.pid})")
                return True
        
        # Clean up any existing server
        if self.server_process is not None:
            print(f"🧹 [LlamaWrapper #{self.instance_id}] Cleaning up old server")
            self._stop_server()
        
        # Prefer existing port (or 8080), but fall back if it's stuck.
        preferred_port = self.server_port or 8080
        self.server_port = self._pick_free_port(preferred_port)
        if not self._ensure_port_free(self.server_port, wait_seconds=5.0):
            fallback_port = self._pick_free_port(0)
            if fallback_port != self.server_port:
                self.server_port = fallback_port
                print(
                    f"⚠️  [LlamaWrapper #{self.instance_id}] "
                    f"Port {preferred_port} busy; falling back to {self.server_port}"
                )
            if not self._ensure_port_free(self.server_port, wait_seconds=5.0):
                raise RuntimeError(f"llama-server port {self.server_port} is still in use")
        
        # Build command for llama-server
        cmd = self._build_server_command(model_path)
        
        print(f"🚀 [LlamaWrapper #{self.instance_id}] Starting server: {' '.join(cmd)}")
        
        launch_env = self._build_server_env()
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                universal_newlines=True,
                env=launch_env,
                cwd=str(self.llama_cpp_path.parent),
                **background_process_kwargs(new_process_group=True),
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
            try:
                if self.server_process.poll() is not None:
                    _, stderr_output = self.server_process.communicate(timeout=1)
                    if stderr_output:
                        print(f"stderr: {stderr_output.strip()}")
                    print(
                        f"❌ [LlamaWrapper #{self.instance_id}] "
                        f"llama-server exited with code {self.server_process.returncode}"
                    )
                elif self.server_process.stderr is not None:
                    stderr_output = self.server_process.stderr.read()
                    if stderr_output:
                        print(f"stderr: {stderr_output.strip()}")
            except Exception:
                pass
            self._stop_server()
            return False
    
    def _is_port_free(self, port: int) -> bool:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _pick_free_port(self, preferred_port: int) -> int:
        """Pick a free local port, preferring the requested one if available."""
        if preferred_port and self._is_port_free(preferred_port):
            return preferred_port
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
            sock.close()
            return port
        except Exception:
            return preferred_port

    def _ensure_port_free(self, port: int, wait_seconds: float = 3.0) -> bool:
        if self._is_port_free(port):
            return True

        print(f"⚠️  [LlamaWrapper #{self.instance_id}] Port {port} is in use; attempting cleanup")
        # Try graceful shutdown on any existing server listening there.
        try:
            requests.post(f'http://127.0.0.1:{port}/shutdown', timeout=1)
        except Exception:
            pass

        # If we own a process, terminate it.
        if self.server_process and self.server_process.poll() is None:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=1)
            except Exception:
                try:
                    self.server_process.kill()
                    self.server_process.wait(timeout=1)
                except Exception:
                    pass

        # Wait for port release.
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            if self._is_port_free(port):
                return True
            time.sleep(0.1)

        if self._is_port_free(port):
            return True

        # Last resort: kill any llama-server still holding the port.
        if self._kill_port_holder(port):
            time.sleep(0.2)
            return self._is_port_free(port)

        # Fallback: kill any remaining llama-server processes (port detection may fail).
        if self._kill_all_llama_servers():
            time.sleep(0.2)
            return self._is_port_free(port)

        return self._is_port_free(port)

    def _pids_listening_on_port(self, port: int) -> list:
        """Return PIDs with a LISTEN socket on the given TCP port (best-effort)."""
        inodes = set()
        try:
            with open("/proc/net/tcp", "r") as f:
                next(f, None)  # skip header
                for line in f:
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    local_addr = parts[1]
                    state = parts[3]
                    inode = parts[9]
                    if state != "0A":  # LISTEN
                        continue
                    ip_hex, port_hex = local_addr.split(":")
                    if int(port_hex, 16) == port:
                        inodes.add(inode)
        except Exception:
            return []

        if not inodes:
            return []

        pids = []
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = int(entry)
                fd_dir = Path(f"/proc/{pid}/fd")
                if not fd_dir.exists():
                    continue
                try:
                    for fd in fd_dir.iterdir():
                        try:
                            target = os.readlink(fd)
                        except Exception:
                            continue
                        if target.startswith("socket:[") and target[8:-1] in inodes:
                            pids.append(pid)
                            break
                except Exception:
                    continue
        except Exception:
            return []
        return pids

    def _kill_port_holder(self, port: int) -> bool:
        """Best-effort kill of a llama-server process holding the given port."""
        pids = []
        try:
            import shutil
            if shutil.which("lsof"):
                result = subprocess.run(
                    ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
            elif shutil.which("ss"):
                result = subprocess.run(
                    ["ss", "-lptn", f"sport = :{port}"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                for line in result.stdout.splitlines():
                    if "pid=" in line:
                        # Extract pid=1234
                        parts = line.split("pid=")[1].split(",", 1)
                        pid_str = parts[0].strip()
                        if pid_str.isdigit():
                            pids.append(int(pid_str))
        except Exception:
            pids = []

        if not pids:
            pids = self._pids_listening_on_port(port)

        if not pids:
            return False

        killed = False
        for pid in pids:
            try:
                cmdline_path = Path(f"/proc/{pid}/cmdline")
                if cmdline_path.exists():
                    cmdline = cmdline_path.read_text(errors="ignore")
                    if "llama-server" not in cmdline:
                        continue
                # Terminate then kill if needed.
                os.kill(pid, 15)
                time.sleep(0.2)
                try:
                    os.kill(pid, 0)
                    os.kill(pid, 9)
                except Exception:
                    pass
                killed = True
            except Exception:
                continue
        return killed

    def _kill_all_llama_servers(self) -> bool:
        """Kill any llama-server processes (best-effort)."""
        pids = []
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                cmdline_path = Path(f"/proc/{entry}/cmdline")
                if not cmdline_path.exists():
                    continue
                cmdline = cmdline_path.read_text(errors="ignore")
                if "llama-server" in cmdline:
                    pids.append(int(entry))
        except Exception:
            return False

        if not pids:
            return False

        killed = False
        for pid in pids:
            try:
                os.kill(pid, 15)
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                    os.kill(pid, 9)
                except Exception:
                    pass
                killed = True
            except Exception:
                continue
        return killed
    
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
                        # Kill whole process group if possible (llama-server may fork)
                        try:
                            os.killpg(self.server_process.pid, signal.SIGKILL)
                        except Exception:
                            self.server_process.kill()
                        self.server_process.wait()
            except Exception as e:
                print(f"⚠️  [LlamaWrapper #{self.instance_id}] Error stopping server: {e}")
            finally:
                # Ensure the process is actually gone
                try:
                    if self.server_process and self.server_process.poll() is None:
                        deadline = time.time() + 5.0
                        while time.time() < deadline and self.server_process.poll() is None:
                            time.sleep(0.1)
                        if self.server_process.poll() is None:
                            try:
                                os.killpg(self.server_process.pid, signal.SIGKILL)
                            except Exception:
                                self.server_process.kill()
                    # If the port is still held, try killing the process group (server may have forked).
                    if self.server_process and not self._is_port_free(self.server_port):
                        try:
                            os.killpg(self.server_process.pid, signal.SIGTERM)
                            time.sleep(0.2)
                            if not self._is_port_free(self.server_port):
                                os.killpg(self.server_process.pid, signal.SIGKILL)
                        except Exception:
                            pass
                except Exception:
                    pass
                self.server_process = None
                self.current_model = None
                self.server_ready = False
                # Wait for port to be released (shutdown can be slow)
                self._ensure_port_free(self.server_port, wait_seconds=10.0)
        else:
            # No tracked process; still try to free the port if something else is holding it.
            self._ensure_port_free(self.server_port, wait_seconds=10.0)
    
    def generate_streaming(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        callback: Optional[Callable[[str], None]] = None,
        messages: list = None,
        tools: Optional[list] = None,
        tool_executor: Optional[Callable[[dict], dict]] = None,
        max_tool_rounds: int = 3,
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
            tools: Optional OpenAI-style tool list for pre-response tool resolution.
            tool_executor: Callable that executes tool arguments and returns a dict.
            max_tool_rounds: Maximum tool call rounds before final answer stream.

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
            web_results_used = 0
            web_sources = []

            # Use caller-supplied history; fall back to bare prompt if not provided
            chat_messages = messages if messages else [{"role": "user", "content": prompt}]
            chat_messages = self._trim_messages(chat_messages, self.context_size, max_tokens)

            url = f'http://127.0.0.1:{self.server_port}/v1/chat/completions'
            # Clamp output tokens to leave room for prompt in context.
            prompt_reserve = 128
            effective_max_tokens = max(
                1,
                min(max_tokens, self.context_size - prompt_reserve)
            )
            if effective_max_tokens != max_tokens:
                print(
                    f"⚠️  [LlamaWrapper #{self.instance_id}] "
                    f"Clamping max_tokens {max_tokens} -> {effective_max_tokens} "
                    f"(context_size={self.context_size})"
                )

            if tools and callable(tool_executor):
                chat_messages, web_results_used, web_sources = self._resolve_tool_calls(
                    url=url,
                    chat_messages=chat_messages,
                    max_tokens=effective_max_tokens,
                    temperature=temperature,
                    tools=tools,
                    tool_executor=tool_executor,
                    max_rounds=max_tool_rounds,
                )

            payload = {
                "messages": chat_messages,
                "stream": True,
                "max_tokens": effective_max_tokens,
                "temperature": temperature,
                # Ask for token usage in the final streaming chunk when supported.
                "stream_options": {"include_usage": True},
            }

            response = requests.post(url, json=payload, stream=True, timeout=self.request_timeout)
            self._active_response = response
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
                prompt_chars = sum(self._content_length(m.get("content", "")) for m in chat_messages)
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
                "web_results_used": web_results_used,
                "web_sources": web_sources,
            }
            prompt_tps_text = f"{prompt_tps:.1f}" if prompt_tps is not None else "n/a"
            generation_tps_text = f"{generation_tps:.1f}" if generation_tps is not None else "n/a"

            print(
                f"✅ [LlamaWrapper #{self.instance_id}] Response complete ({len(full_response)} chars) "
                f"[prompt {prompt_tps_text} t/s | generation {generation_tps_text} t/s]"
            )

        except Exception as e:
            print(f"❌ [LlamaWrapper #{self.instance_id}] Exception during generation: {e}")
            if self.stop_requested:
                return
            raise
        finally:
            self.is_generating = False
            if self._active_response is not None:
                try:
                    self._active_response.close()
                except Exception:
                    pass
                self._active_response = None

    @staticmethod
    def _content_length(content) -> int:
        if isinstance(content, str):
            return len(content)
        if content is None:
            return 0
        try:
            return len(json.dumps(content, ensure_ascii=False))
        except Exception:
            return len(str(content))

    @staticmethod
    def _parse_tool_arguments(raw_args) -> dict:
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            text = raw_args.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _extract_text_tool_calls(content) -> list:
        text = str(content or "")
        if not text:
            return []

        calls = []
        pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
        for idx, match in enumerate(pattern.finditer(text), start=1):
            payload_text = match.group(1).strip()
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            name = str(payload.get("name", "")).strip()
            if not name:
                continue
            args = payload.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append(
                {
                    "id": f"text_tool_call_{idx}",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            )
        return calls

    @staticmethod
    def _merge_web_sources(existing: list, tool_result: dict, limit: int = 5) -> list:
        merged = list(existing or [])
        seen_urls = {str(item.get("url", "")).strip() for item in merged if isinstance(item, dict)}

        for entry in (tool_result or {}).get("results", []) or []:
            if len(merged) >= limit:
                break
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            merged.append(
                {
                    "title": str(entry.get("title", "")).strip() or url,
                    "url": url,
                }
            )

        return merged[:limit]

    def _resolve_tool_calls(
        self,
        url: str,
        chat_messages: list,
        max_tokens: int,
        temperature: float,
        tools: list,
        tool_executor: Callable[[dict], dict],
        max_rounds: int = 3,
    ) -> tuple:
        resolved_messages = list(chat_messages)
        web_results_used = 0
        web_sources = []

        for _ in range(max(1, int(max_rounds))):
            response = requests.post(
                url,
                json={
                    "messages": resolved_messages,
                    "stream": False,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "tools": tools,
                    "tool_choice": "auto",
                },
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            choices = payload.get("choices") or []
            if not choices:
                break

            message = choices[0].get("message") or {}
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                tool_calls = self._extract_text_tool_calls(message.get("content", ""))
            if not tool_calls:
                break

            resolved_messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "tool_calls": tool_calls,
                }
            )

            for tool_call in tool_calls:
                function_payload = tool_call.get("function") or {}
                tool_name = function_payload.get("name")
                tool_args = self._parse_tool_arguments(function_payload.get("arguments"))
                tool_result = tool_executor(tool_args)
                if tool_name == "internet_search":
                    web_results_used += 1
                    web_sources = self._merge_web_sources(web_sources, tool_result, limit=5)

                tool_message = {
                    "role": "tool",
                    "name": tool_name or "internet_search",
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
                tool_call_id = tool_call.get("id")
                if tool_call_id:
                    tool_message["tool_call_id"] = tool_call_id
                resolved_messages.append(tool_message)

        if web_results_used > 0:
            resolved_messages.append(
                {
                    "role": "system",
                    "content": (
                        "You already have internet search results. "
                        "Provide the final answer now in plain text. "
                        "Do not output <tool_call> tags or request more tools."
                    ),
                }
            )
        return resolved_messages, web_results_used, web_sources

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
        self.stop_requested = True
        self.is_generating = False
        if self._active_response is not None:
            try:
                self._active_response.close()
            except Exception:
                pass
            self._active_response = None
        self._stop_server()

    def preload_model(self, model_path: str) -> bool:
        """Stop current server, start with the new model, and run a tiny warmup."""
        try:
            # Always restart to ensure a clean switch.
            if self.server_process is not None:
                self._stop_server()
            # Give the OS time to release the port, then retry start a few times.
            preferred_port = self.server_port or 8080
            if not self._ensure_port_free(preferred_port, wait_seconds=8.0):
                # Fall back to a free port if 8080 stays stuck.
                self.server_port = self._pick_free_port(preferred_port)
                if self.server_port != preferred_port:
                    print(
                        f"⚠️  [LlamaWrapper #{self.instance_id}] "
                        f"Port {preferred_port} busy; falling back to {self.server_port}"
                    )
                if not self._ensure_port_free(self.server_port, wait_seconds=5.0):
                    raise RuntimeError(f"llama-server port {self.server_port} is still in use")

            for attempt in range(3):
                if self._start_server(model_path):
                    break
                if attempt < 2:
                    time.sleep(0.5)
            if self.server_process is None or self.server_process.poll() is not None:
                return False

            # Warmup: tiny non-streaming request to load weights.
            url = f'http://127.0.0.1:{self.server_port}/v1/chat/completions'
            payload = {
                "messages": [{"role": "user", "content": "warmup"}],
                "stream": False,
                "max_tokens": 1,
                "temperature": 0.0,
            }
            try:
                requests.post(url, json=payload, timeout=min(15, self.request_timeout))
            except Exception:
                # Warmup is best-effort; ignore failures.
                pass
            return True
        except Exception as e:
            print(f"⚠️  [LlamaWrapper #{self.instance_id}] Preload failed: {e}")
            return False

    def _trim_messages(self, messages, context_size: int, max_tokens: int):
        """
        Approximate trim to fit within context window.
        Estimate 1 token ~= 4 chars and preserve system prompt when present.
        """
        if not messages:
            return messages
        budget_tokens = max(256, context_size - max_tokens)
        budget_chars = budget_tokens * 4

        trimmed = []
        start_idx = 0
        if messages[0].get("role") == "system":
            trimmed.append(messages[0])
            start_idx = 1
            budget_chars -= self._content_length(messages[0].get("content", ""))

        for msg in reversed(messages[start_idx:]):
            content = msg.get("content", "")
            cost = self._content_length(content)
            if budget_chars - cost < 0:
                break
            trimmed.append(msg)
            budget_chars -= cost

        if start_idx == 1:
            return [messages[0]] + list(reversed(trimmed[1:]))
        return list(reversed(trimmed))
    
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
