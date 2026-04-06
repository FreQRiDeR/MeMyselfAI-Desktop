"""
unified_backend.py
Unified backend supporting local llama.cpp, remote llama-server, Ollama, and HuggingFace
"""

import sys
import json
import re
import requests
import subprocess
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Generator, Optional, Callable
from html import unescape
from pathlib import Path
from urllib.parse import quote
from backend.process_utils import background_process_kwargs


class BackendType(Enum):
    """Available backend types"""
    LOCAL = "local"           # Local llama.cpp
    LLAMA_SERVER = "llama_server"  # HTTP + SSE
    OLLAMA = "ollama"         # Ollama API (local or remote)
    HUGGINGFACE = "huggingface"  # HuggingFace Inference API


class UnifiedBackend:
    """Unified interface for multiple LLM backends"""

    OLLAMA_CLOUD_URL = 'https://ollama.com/api'
    
    def __init__(self, backend_type: BackendType, **config):
        """
        Initialize backend
        
        Args:
            backend_type: Type of backend to use
            **config: Backend-specific configuration
                For LOCAL: llama_cpp_path
                For LLAMA_SERVER: llama_server_url, llama_server_api_key
                For OLLAMA: ollama_url (default: http://localhost:11434), ollama_path
                For HUGGINGFACE: api_key
        """
        self.backend_type = backend_type
        self.config = config
        self.inference_timeout = int(config.get('inference_timeout', 300))
        self.ollama_process = None  # To track Ollama process
        self.last_generation_stats = {}
        self._active_response = None
        
        # Initialize backend-specific components
        if backend_type == BackendType.LOCAL:
            from backend.llama_wrapper import LlamaWrapper
            llama_path = config.get('llama_cpp_path', 'bundled')
            self.local_wrapper = LlamaWrapper(llama_path, tuning=config)
        elif backend_type == BackendType.LLAMA_SERVER:
            self.llama_server_url = config.get('llama_server_url', 'http://localhost:8080')
            self.llama_server_api_key = str(config.get('llama_server_api_key', '')).strip()
        elif backend_type == BackendType.OLLAMA:
            self.ollama_url = config.get('ollama_url', 'http://localhost:11434')
            self.ollama_path = config.get('ollama_path', 'bundled')
            self.ollama_api_key = str(config.get('ollama_api_key', '')).strip()
            self.ollama_cloud_url = config.get('ollama_cloud_url', self.OLLAMA_CLOUD_URL)
            self._start_ollama_if_needed()
        elif backend_type == BackendType.HUGGINGFACE:
            self.hf_api_key = config.get('api_key')
            if not self.hf_api_key:
                raise ValueError("HuggingFace API key required")

    @staticmethod
    def _ollama_headers(api_key: str = "") -> dict:
        headers = {}
        api_key = str(api_key).strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @staticmethod
    def _ollama_api_url(base_url: str, path: str) -> str:
        base = str(base_url).rstrip('/')
        suffix = path.lstrip('/')
        if base.endswith('/api'):
            return f'{base}/{suffix}'
        return f'{base}/api/{suffix}'

    @staticmethod
    def _llama_server_headers(api_key: str = "") -> dict:
        headers = {}
        api_key = str(api_key).strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @classmethod
    def _llama_server_base_url(cls, base_url: str) -> str:
        base = str(base_url).strip().rstrip('/')
        for suffix in ('/v1/chat/completions', '/v1/completions', '/v1', '/health'):
            if base.endswith(suffix):
                return base[:-len(suffix)].rstrip('/')
        return base

    @classmethod
    def _llama_server_chat_url(cls, base_url: str) -> str:
        return f"{cls._llama_server_base_url(base_url)}/v1/chat/completions"

    @classmethod
    def _llama_server_health_url(cls, base_url: str) -> str:
        return f"{cls._llama_server_base_url(base_url)}/health"

    @staticmethod
    def _normalize_cloud_model_name(model_name: str) -> str:
        if model_name.endswith(':cloud'):
            return model_name[:-6]
        if model_name.endswith('-cloud'):
            return model_name[:-6]
        return model_name

    @classmethod
    def _is_cloud_model_name(cls, model_name: str) -> bool:
        model_name = str(model_name).strip().lower()
        return model_name.endswith(':cloud') or model_name.endswith('-cloud')

    def _resolve_ollama_target(self, model) -> dict:
        if isinstance(model, dict):
            route = str(model.get('route', 'local')).lower()
            request_model = model.get('request_model') or model.get('name') or ''
            display_name = model.get('display_name') or model.get('name') or request_model
        else:
            display_name = str(model)
            route = 'cloud' if self._is_cloud_model_name(display_name) else 'local'
            request_model = (
                self._normalize_cloud_model_name(display_name)
                if route == 'cloud'
                else display_name
            )

        base_url = self.ollama_cloud_url if route == 'cloud' else self.ollama_url
        headers = self._ollama_headers(self.ollama_api_key if route == 'cloud' else '')

        return {
            'route': route,
            'request_model': request_model,
            'display_name': display_name,
            'base_url': base_url,
            'headers': headers,
        }

    @staticmethod
    def _message_content_length(content) -> int:
        if isinstance(content, str):
            return len(content)
        if content is None:
            return 0
        try:
            return len(json.dumps(content, ensure_ascii=False))
        except Exception:
            return len(str(content))

    @staticmethod
    def _clamp_int(value, default: int, min_value: int, max_value: int) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError):
            result = default
        return max(min_value, min(max_value, result))

    @staticmethod
    def _internet_tool_spec() -> dict:
        return {
            "type": "function",
            "function": {
                "name": "internet_search",
                "description": (
                    "Search the internet for up-to-date information and return concise "
                    "results with source URLs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to look up on the internet."
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Number of results to return (1-8).",
                            "minimum": 1,
                            "maximum": 8
                        }
                    },
                    "required": ["query"]
                }
            }
        }

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
    def _force_final_answer_instruction() -> str:
        return (
            "You already have internet search results. "
            "Provide the final answer now in plain text. "
            "Do not output <tool_call> tags or request more tools."
        )

    def _run_internet_tool(self, args: dict) -> dict:
        query = str((args or {}).get("query", "")).strip()
        max_results = self._clamp_int((args or {}).get("max_results", 5), default=5, min_value=1, max_value=8)
        return self._internet_search(query=query, max_results=max_results)

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

    @staticmethod
    def _extract_text_content(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text = str(part.get("text", "")).strip()
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
        return str(content or "").strip()

    def _latest_user_query(self, chat_messages: list, prompt: str = "") -> str:
        for msg in reversed(chat_messages or []):
            if str(msg.get("role", "")).lower() != "user":
                continue
            text = self._extract_text_content(msg.get("content", ""))
            if text:
                return text
        return str(prompt or "").strip()

    @staticmethod
    def _should_force_web_search(query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        triggers = (
            "internet", "web", "search", "browse", "look up", "lookup",
            "latest", "current", "today", "verify", "fact-check", "fact check",
            "check your results", "news",
        )
        return any(token in text for token in triggers)

    def _build_web_context_message(self, tool_result: dict) -> str:
        query = str((tool_result or {}).get("query", "")).strip()
        fetched_at = str((tool_result or {}).get("fetched_at", "")).strip()
        results = (tool_result or {}).get("results", []) or []
        error = str((tool_result or {}).get("error", "")).strip()
        lines = [
            "INTERNET_SEARCH_RESULTS (auto-fetched by app):",
            f"Query: {query}" if query else "Query: (empty)",
        ]
        if fetched_at:
            lines.append(f"Fetched at: {fetched_at}")
        if results:
            lines.append("Top results:")
            for idx, entry in enumerate(results[:5], start=1):
                title = str(entry.get("title", "")).strip() or "(untitled)"
                url = str(entry.get("url", "")).strip() or "(no url)"
                snippet = str(entry.get("snippet", "")).strip()
                lines.append(f"{idx}. {title}")
                lines.append(f"   URL: {url}")
                if snippet:
                    lines.append(f"   Snippet: {snippet[:320]}")
        elif error:
            lines.append(f"Search error: {error}")

        lines.append(
            "Use these web results in your next answer. If citing facts, include source URLs."
        )
        return "\n".join(lines)

    def _apply_forced_web_context_if_needed(
        self,
        chat_messages: list,
        prompt: str,
        internet_enabled: bool,
        web_results_used: int,
        web_sources: list = None,
    ) -> tuple:
        if not internet_enabled or web_results_used > 0:
            return chat_messages, web_results_used, list(web_sources or [])

        query = self._latest_user_query(chat_messages, prompt=prompt)
        if not self._should_force_web_search(query):
            return chat_messages, web_results_used, list(web_sources or [])

        tool_result = self._internet_search(query=query, max_results=5)
        context_msg = self._build_web_context_message(tool_result)
        updated_messages = list(chat_messages)
        updated_messages.append({"role": "system", "content": context_msg})
        merged_sources = self._merge_web_sources(web_sources or [], tool_result, limit=5)

        if (tool_result.get("results") or []):
            web_results_used += 1
        return updated_messages, web_results_used, merged_sources

    def _internet_search(self, query: str, max_results: int = 5) -> dict:
        query = str(query or "").strip()
        max_results = self._clamp_int(max_results, default=5, min_value=1, max_value=8)
        if not query:
            return {
                "query": query,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "results": [],
                "error": "Empty query"
            }

        headers = {"User-Agent": "MeMyselfAI/1.0"}
        results = []
        errors = []
        seen_urls = set()

        def add_result(title: str, url: str, snippet: str):
            if len(results) >= max_results:
                return
            clean_url = str(url or "").strip()
            if not clean_url or clean_url in seen_urls:
                return
            seen_urls.add(clean_url)
            results.append({
                "title": str(title or "").strip()[:180] or clean_url,
                "url": clean_url,
                "snippet": str(snippet or "").strip()[:500]
            })

        def add_duck_topics(items):
            if not isinstance(items, list):
                return
            for item in items:
                if len(results) >= max_results:
                    return
                if not isinstance(item, dict):
                    continue
                nested = item.get("Topics")
                if isinstance(nested, list):
                    add_duck_topics(nested)
                text = str(item.get("Text", "")).strip()
                url = str(item.get("FirstURL", "")).strip()
                if not text or not url:
                    continue
                title = text.split(" - ", 1)[0]
                add_result(title, url, text)

        try:
            response = requests.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": 1,
                    "no_redirect": 1,
                    "skip_disambig": 1,
                },
                headers=headers,
                timeout=min(12, self.inference_timeout),
            )
            response.raise_for_status()
            payload = response.json()
            abstract = str(payload.get("AbstractText", "")).strip()
            if abstract:
                heading = str(payload.get("Heading", "")).strip() or query
                abstract_url = str(payload.get("AbstractURL", "")).strip()
                if abstract_url:
                    add_result(heading, abstract_url, abstract)
            add_duck_topics(payload.get("RelatedTopics", []))
        except Exception as exc:
            errors.append(f"DuckDuckGo: {exc}")

        # Fallback source for broader recall when DDG Instant has sparse results.
        if len(results) < max_results:
            try:
                remaining = max_results - len(results)
                response = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "srlimit": remaining,
                        "utf8": 1,
                        "format": "json",
                    },
                    headers=headers,
                    timeout=min(12, self.inference_timeout),
                )
                response.raise_for_status()
                wiki_payload = response.json()
                for entry in wiki_payload.get("query", {}).get("search", []):
                    title = str(entry.get("title", "")).strip()
                    if not title:
                        continue
                    snippet_html = str(entry.get("snippet", ""))
                    snippet = unescape(re.sub(r"<[^>]+>", "", snippet_html))
                    url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                    add_result(title, url, snippet)
            except Exception as exc:
                errors.append(f"Wikipedia: {exc}")

        result_payload = {
            "query": query,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "results": results[:max_results],
        }
        if errors and not results:
            result_payload["error"] = " | ".join(errors)
        elif errors:
            result_payload["warnings"] = errors
        return result_payload

    def _resolve_llama_server_internet_tools(
        self,
        chat_messages: list,
        max_tokens: int,
        temperature: float,
        max_rounds: int = 3,
    ) -> tuple:
        resolved_messages = list(chat_messages)
        tool_spec = [self._internet_tool_spec()]
        web_results_used = 0
        web_sources = []

        for _ in range(max_rounds):
            response = requests.post(
                self._llama_server_chat_url(self.llama_server_url),
                headers=self._llama_server_headers(self.llama_server_api_key),
                json={
                    "messages": resolved_messages,
                    "stream": False,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "tools": tool_spec,
                    "tool_choice": "auto",
                },
                timeout=self.inference_timeout,
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

            assistant_message = {
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": tool_calls,
            }
            resolved_messages.append(assistant_message)

            for call in tool_calls:
                function_payload = call.get("function") or {}
                tool_name = function_payload.get("name")
                tool_args = self._parse_tool_arguments(function_payload.get("arguments"))

                if tool_name == "internet_search":
                    tool_result = self._run_internet_tool(tool_args)
                    web_results_used += 1
                    web_sources = self._merge_web_sources(web_sources, tool_result, limit=5)
                else:
                    tool_result = {"error": f"Unsupported tool: {tool_name}"}

                tool_message = {
                    "role": "tool",
                    "name": tool_name or "internet_search",
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
                tool_call_id = call.get("id")
                if tool_call_id:
                    tool_message["tool_call_id"] = tool_call_id
                resolved_messages.append(tool_message)

        if web_results_used > 0:
            resolved_messages.append(
                {"role": "system", "content": self._force_final_answer_instruction()}
            )
        return resolved_messages, web_results_used, web_sources

    def _resolve_ollama_internet_tools(
        self,
        target: dict,
        chat_messages: list,
        max_tokens: int,
        temperature: float,
        max_rounds: int = 3,
    ) -> tuple:
        resolved_messages = list(chat_messages)
        tool_spec = [self._internet_tool_spec()]
        web_results_used = 0
        web_sources = []

        for _ in range(max_rounds):
            response = requests.post(
                self._ollama_api_url(target['base_url'], 'chat'),
                headers=target['headers'],
                json={
                    "model": target['request_model'],
                    "messages": resolved_messages,
                    "stream": False,
                    "tools": tool_spec,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
                timeout=self.inference_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            message = payload.get("message") or {}
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                tool_calls = self._extract_text_tool_calls(message.get("content", ""))
            if not tool_calls:
                break

            assistant_message = {
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": tool_calls,
            }
            resolved_messages.append(assistant_message)

            for call in tool_calls:
                function_payload = call.get("function") or {}
                tool_name = function_payload.get("name")
                tool_args = self._parse_tool_arguments(function_payload.get("arguments"))

                if tool_name == "internet_search":
                    tool_result = self._run_internet_tool(tool_args)
                    web_results_used += 1
                    web_sources = self._merge_web_sources(web_sources, tool_result, limit=5)
                else:
                    tool_result = {"error": f"Unsupported tool: {tool_name}"}

                resolved_messages.append(
                    {
                        "role": "tool",
                        "name": tool_name or "internet_search",
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        if web_results_used > 0:
            resolved_messages.append(
                {"role": "system", "content": self._force_final_answer_instruction()}
            )
        return resolved_messages, web_results_used, web_sources
    
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
                        stderr=subprocess.DEVNULL,
                        **background_process_kwargs(),
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
                Path(bundle_dir) / 'backend' / 'bin' / 'linux' / 'ollama',
                Path(bundle_dir) / 'ollama',
            ]
        else:
            # Development mode
            possible_paths = [
                Path(__file__).parent / 'bin' / 'ollama',
                Path(__file__).parent / 'bin' / 'linux' / 'ollama',
                Path('./backend/bin/linux/ollama'),
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
        messages: list = None,
        internet_enabled: bool = False,
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
            internet_enabled: Enable internet search tool-calling protocol.

        Yields:
            Generated tokens as they arrive
        """
        if self.backend_type == BackendType.LOCAL:
            yield from self._local_generate(
                model, prompt, max_tokens, temperature, callback, messages, internet_enabled
            )
        elif self.backend_type == BackendType.LLAMA_SERVER:
            yield from self._llama_server_generate(
                model, prompt, max_tokens, temperature, callback, messages, internet_enabled
            )
        elif self.backend_type == BackendType.OLLAMA:
            yield from self._ollama_generate(
                model, prompt, max_tokens, temperature, callback, messages, internet_enabled
            )
        elif self.backend_type == BackendType.HUGGINGFACE:
            yield from self._hf_generate(model, prompt, max_tokens, temperature, callback)
    
    def _local_generate(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        callback: Optional[Callable[[str], None]],
        messages: list = None,
        internet_enabled: bool = False,
    ) -> Generator[str, None, None]:
        """Generate using local llama.cpp"""
        chat_messages = messages if messages else [{"role": "user", "content": prompt}]
        forced_web_results_used = 0
        forced_web_sources = []
        if internet_enabled:
            chat_messages, forced_web_results_used, forced_web_sources = self._apply_forced_web_context_if_needed(
                chat_messages,
                prompt=prompt,
                internet_enabled=internet_enabled,
                web_results_used=0,
                web_sources=[],
            )

        tools = [self._internet_tool_spec()] if internet_enabled else None
        yield from self.local_wrapper.generate_streaming(
            model_path,
            prompt,
            max_tokens,
            temperature,
            callback,
            chat_messages,
            tools=tools,
            tool_executor=self._run_internet_tool if internet_enabled else None,
        )

        if forced_web_results_used > 0 and hasattr(self.local_wrapper, "last_generation_stats"):
            stats = dict(self.local_wrapper.get_last_generation_stats() or {})
            stats["web_results_used"] = int(stats.get("web_results_used", 0) or 0) + forced_web_results_used
            stats["web_sources"] = self._merge_web_sources(
                stats.get("web_sources", []) or [],
                {"results": forced_web_sources},
                limit=5,
            )
            self.local_wrapper.last_generation_stats = stats

    def _llama_server_generate(
        self,
        model,
        prompt: str,
        max_tokens: int,
        temperature: float,
        callback: Optional[Callable[[str], None]],
        messages: list = None,
        internet_enabled: bool = False,
    ) -> Generator[str, None, None]:
        """Generate using a remote llama-server over HTTP + SSE."""
        try:
            request_start = time.time()
            first_token_time = None
            stream_usage = None
            full_response = ""
            web_results_used = 0
            web_sources = []

            chat_messages = messages if messages else [{"role": "user", "content": prompt}]
            if internet_enabled:
                chat_messages, web_results_used, web_sources = self._resolve_llama_server_internet_tools(
                    chat_messages, max_tokens=max_tokens, temperature=temperature
                )
            chat_messages, web_results_used, web_sources = self._apply_forced_web_context_if_needed(
                chat_messages,
                prompt=prompt,
                internet_enabled=internet_enabled,
                web_results_used=web_results_used,
                web_sources=web_sources,
            )
            payload = {
                "messages": chat_messages,
                "stream": True,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream_options": {"include_usage": True},
            }

            response = requests.post(
                self._llama_server_chat_url(self.llama_server_url),
                headers=self._llama_server_headers(self.llama_server_api_key),
                json=payload,
                stream=True,
                timeout=self.inference_timeout,
            )
            self._active_response = response
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode('utf-8') if isinstance(line, bytes) else str(line)
                if not line_str.startswith('data: '):
                    continue

                data_str = line_str[6:]
                if data_str.strip() == '[DONE]':
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if isinstance(chunk, dict) and "usage" in chunk:
                    stream_usage = chunk.get("usage") or stream_usage

                choices = chunk.get('choices') or []
                if not choices:
                    continue

                delta = choices[0].get('delta', {})
                content = delta.get('content', '')
                if not content:
                    continue

                if first_token_time is None:
                    first_token_time = time.time()
                full_response += content
                if callback:
                    callback(content)
                yield content

            request_end = time.time()
            if first_token_time is not None:
                prompt_seconds = max(1e-9, first_token_time - request_start)
                generation_seconds = max(1e-9, request_end - first_token_time)
            else:
                prompt_seconds = max(1e-9, request_end - request_start)
                generation_seconds = None

            prompt_tokens = None
            completion_tokens = None
            if isinstance(stream_usage, dict):
                prompt_tokens = stream_usage.get("prompt_tokens")
                completion_tokens = stream_usage.get("completion_tokens")

            if prompt_tokens is None:
                prompt_chars = sum(
                    self._message_content_length(m.get("content", "")) for m in chat_messages
                )
                prompt_tokens = max(1, prompt_chars // 4)
            if completion_tokens is None:
                completion_tokens = max(1, len(full_response) // 4)

            prompt_tps = prompt_tokens / prompt_seconds if prompt_seconds and prompt_tokens else None
            generation_tps = (
                completion_tokens / generation_seconds
                if generation_seconds and completion_tokens else None
            )

            self.last_generation_stats = {
                "prompt_tps": prompt_tps,
                "generation_tps": generation_tps,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "web_results_used": web_results_used,
                "web_sources": web_sources,
            }
        finally:
            if self._active_response is not None:
                try:
                    self._active_response.close()
                except Exception:
                    pass
                self._active_response = None
    
    def _ollama_generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        callback: Optional[Callable[[str], None]],
        messages: list = None,
        internet_enabled: bool = False,
    ) -> Generator[str, None, None]:
        """Generate using Ollama /api/chat with full conversation history"""
        try:
            # Use provided history, or wrap the bare prompt as a single user turn
            chat_messages = messages if messages else [{"role": "user", "content": prompt}]
            target = self._resolve_ollama_target(model)
            web_results_used = 0
            web_sources = []
            if internet_enabled:
                chat_messages, web_results_used, web_sources = self._resolve_ollama_internet_tools(
                    target, chat_messages, max_tokens=max_tokens, temperature=temperature
                )
            chat_messages, web_results_used, web_sources = self._apply_forced_web_context_if_needed(
                chat_messages,
                prompt=prompt,
                internet_enabled=internet_enabled,
                web_results_used=web_results_used,
                web_sources=web_sources,
            )
            request_start = time.time()
            first_token_time = None
            full_response = ""
            final_chunk = None

            response = requests.post(
                self._ollama_api_url(target['base_url'], 'chat'),
                headers=target['headers'],
                json={
                    "model": target['request_model'],
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
            self._active_response = response
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if data.get('done', False):
                            final_chunk = data
                            continue
                        if not data.get('done', False):
                            # /api/chat returns tokens under message.content
                            token = data.get('message', {}).get('content', '')
                            if token:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                full_response += token
                                if callback:
                                    callback(token)
                                yield token
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
            if isinstance(final_chunk, dict):
                prompt_tokens = final_chunk.get("prompt_eval_count") or final_chunk.get("prompt_tokens")
                completion_tokens = final_chunk.get("eval_count") or final_chunk.get("completion_tokens")
                prompt_eval_duration = final_chunk.get("prompt_eval_duration")
                eval_duration = final_chunk.get("eval_duration")
                # Ollama reports nanoseconds for these durations.
                if prompt_eval_duration:
                    prompt_seconds = max(1e-9, prompt_eval_duration / 1_000_000_000)
                if eval_duration:
                    generation_seconds = max(1e-9, eval_duration / 1_000_000_000)

            if prompt_tokens is None:
                prompt_chars = sum(
                    self._message_content_length(m.get("content", "")) for m in chat_messages
                )
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

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code == 401:
                if 'signin_url' in (e.response.text if e.response is not None else ''):
                    raise RuntimeError(
                        "Ollama Cloud authorization required for this model. "
                        "Set your Ollama API key in Settings or choose a local model."
                    )
                raise RuntimeError("Ollama API error: 401 Unauthorized.")
            raise RuntimeError(f"Ollama API error: {e}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")
        finally:
            if self._active_response is not None:
                try:
                    self._active_response.close()
                except Exception:
                    pass
                self._active_response = None
    
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
        if self.backend_type in {BackendType.LLAMA_SERVER, BackendType.OLLAMA}:
            return dict(self.last_generation_stats)
        return {}
    
    def stop_generation(self):
        """Stop current generation"""
        if self.backend_type == BackendType.LOCAL:
            self.local_wrapper.stop_generation()
        elif self.backend_type in {BackendType.LLAMA_SERVER, BackendType.OLLAMA}:
            if self._active_response is not None:
                try:
                    self._active_response.close()
                except Exception:
                    pass
                self._active_response = None
        # HF doesn't need explicit stopping (HTTP request ends)
    
    def cleanup(self):
        """Clean up backend resources"""
        # Clean up LOCAL llama-server process via the wrapper
        if self.backend_type == BackendType.LOCAL and hasattr(self, 'local_wrapper'):
            try:
                self.local_wrapper.cleanup()
            except Exception as e:
                print(f"⚠️  llama_wrapper cleanup error: {e}")

        if self._active_response is not None:
            try:
                self._active_response.close()
            except Exception:
                pass
            self._active_response = None

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

    def preload_model(self, model_path: str) -> bool:
        """Preload/warm a local llama.cpp model server."""
        if self.backend_type == BackendType.LOCAL and hasattr(self, "local_wrapper"):
            return self.local_wrapper.preload_model(model_path)
        return False
    
    @staticmethod
    def get_ollama_models(ollama_url: str = 'http://localhost:11434', api_key: str = '') -> list:
        """Get list of available Ollama models"""
        try:
            response = requests.get(
                UnifiedBackend._ollama_api_url(ollama_url, 'tags'),
                headers=UnifiedBackend._ollama_headers(api_key),
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            return [m['name'] for m in data.get('models', [])]
        except Exception as e:
            print(f"⚠️  Failed to fetch Ollama models: {e}")
            return []
    
    @staticmethod
    def test_ollama_connection(ollama_url: str = 'http://localhost:11434', api_key: str = '') -> bool:
        """Test if Ollama is running and accessible"""
        try:
            response = requests.get(
                UnifiedBackend._ollama_api_url(ollama_url, 'tags'),
                headers=UnifiedBackend._ollama_headers(api_key),
                timeout=2
            )
            return response.status_code in {200, 401}
        except:
            return False

    @staticmethod
    def test_llama_server_connection(llama_server_url: str = 'http://localhost:8080', api_key: str = '') -> bool:
        """Test if a llama-server endpoint is running and accessible."""
        try:
            response = requests.get(
                UnifiedBackend._llama_server_health_url(llama_server_url),
                headers=UnifiedBackend._llama_server_headers(api_key),
                timeout=2,
            )
            return response.status_code in {200, 401}
        except Exception:
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
