"""
unified_backend.py
Unified backend supporting local llama.cpp, remote llama-server, Ollama,
HuggingFace, and OpenAI-compatible chat completion APIs.
"""

import sys
import json
import re
import requests
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from enum import Enum
from typing import Generator, Optional, Callable
from html import unescape
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from backend.process_utils import background_process_kwargs


class BackendType(Enum):
    """Available backend types"""
    LOCAL = "local"           # Local llama.cpp
    LLAMA_SERVER = "llama_server"  # HTTP + SSE
    OLLAMA = "ollama"         # Ollama API (local or remote)
    HUGGINGFACE = "huggingface"  # HuggingFace Inference API
    OPENAI_COMPATIBLE = "openai_compatible"  # OpenAI-compatible /v1/chat/completions


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
                For OPENAI_COMPATIBLE: openai_base_url, openai_api_key, openai_model
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
            self.llama_server_tool_protocol_supported = None  # None=unknown, True=supported, False=rejected
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
        elif backend_type == BackendType.OPENAI_COMPATIBLE:
            self.openai_base_url = config.get('openai_base_url', 'https://api.openai.com/v1')
            self.openai_api_key = str(config.get('openai_api_key', '')).strip()
            self.openai_model = str(config.get('openai_model', '')).strip()
            self.openai_tool_protocol_supported = None
            if not self.openai_base_url:
                raise ValueError("OpenAI-compatible base URL required")
            if not self.openai_model:
                raise ValueError("OpenAI-compatible model required")

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

    @staticmethod
    def _openai_compatible_headers(api_key: str = "") -> dict:
        headers = {}
        api_key = str(api_key).strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @classmethod
    def _openai_compatible_base_url(cls, base_url: str) -> str:
        base = str(base_url).strip().rstrip('/')
        for suffix in ('/chat/completions', '/completions'):
            if base.endswith(suffix):
                return base[:-len(suffix)].rstrip('/')
        return base

    @classmethod
    def _openai_compatible_chat_url(cls, base_url: str) -> str:
        base = cls._openai_compatible_base_url(base_url)
        # If the base URL already contains a version segment (like /v1, /v1beta),
        # append /chat/completions directly. Otherwise, assume the standard OpenAI /v1 prefix.
        if re.search(r'/v\d+(beta)?$', base):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

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
    def _is_ollama_chat_template_error(status_code: int, response_text: str = "") -> bool:
        text = str(response_text or "").lower()
        if status_code != 400:
            return False
        return (
            'chat template' in text
            or 'no chat template' in text
            or 'does not support chat' in text
            or 'chat_template' in text
        )

    @staticmethod
    def _ollama_prompt_from_messages(messages: list) -> str:
        if not messages:
            return ""
        lines = []
        for message in messages:
            role = str(message.get('role', 'user')).strip().lower()
            content = message.get('content', '')
            if isinstance(content, (list, dict)):
                content = json.dumps(content, ensure_ascii=False)
            content = str(content or '').strip()
            if not content:
                continue
            if role == 'system':
                lines.append(f"System: {content}")
            elif role == 'assistant':
                lines.append(f"Assistant: {content}")
            else:
                lines.append(f"User: {content}")
        return "\n".join(lines).strip()

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

    def _config_float(self, key: str, default: float) -> float:
        try:
            return float(self.config.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def _config_int(self, key: str, default: int) -> int:
        try:
            return int(self.config.get(key, default))
        except (TypeError, ValueError):
            return int(default)

    def _stop_sequences(self) -> list:
        raw = str(self.config.get("stop_sequences", "") or "").strip()
        if not raw:
            return []
        raw = raw.replace("\\n", "\n")
        return [line for line in (part.strip() for part in raw.splitlines()) if line]

    def _api_extra_body(self) -> dict:
        raw = str(self.config.get("api_extra_body", "") or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    @classmethod
    def _deep_merge_dicts(cls, base: dict, extra: dict) -> dict:
        merged = dict(base or {})
        for key, value in (extra or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = cls._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _openai_request_extras(self) -> dict:
        extras = {}
        top_p = self._config_float("top_p", 1.0)
        if 0.0 < top_p <= 1.0:
            extras["top_p"] = top_p

        presence_penalty = self._config_float("presence_penalty", 0.0)
        if presence_penalty != 0.0:
            extras["presence_penalty"] = presence_penalty

        frequency_penalty = self._config_float("frequency_penalty", 0.0)
        if frequency_penalty != 0.0:
            extras["frequency_penalty"] = frequency_penalty

        seed = self._config_int("seed", -1)
        if seed >= 0:
            extras["seed"] = seed

        stop = self._stop_sequences()
        if stop:
            extras["stop"] = stop[0] if len(stop) == 1 else stop
        return extras

    def _llama_server_request_extras(self) -> dict:
        extras = self._openai_request_extras()
        top_k = self._config_int("top_k", 0)
        if top_k > 0:
            extras["top_k"] = top_k

        min_p = self._config_float("min_p", 0.0)
        if min_p > 0.0:
            extras["min_p"] = min_p

        repeat_penalty = self._config_float("repeat_penalty", 1.0)
        if repeat_penalty != 1.0:
            extras["repeat_penalty"] = repeat_penalty
        return extras

    def _ollama_options(self, max_tokens: int, temperature: float) -> dict:
        options = {
            "num_predict": max_tokens,
            "temperature": temperature,
        }

        top_p = self._config_float("top_p", 1.0)
        if 0.0 < top_p <= 1.0:
            options["top_p"] = top_p

        top_k = self._config_int("top_k", 0)
        if top_k > 0:
            options["top_k"] = top_k

        min_p = self._config_float("min_p", 0.0)
        if min_p > 0.0:
            options["min_p"] = min_p

        repeat_penalty = self._config_float("repeat_penalty", 1.0)
        if repeat_penalty != 1.0:
            options["repeat_penalty"] = repeat_penalty

        seed = self._config_int("seed", -1)
        if seed >= 0:
            options["seed"] = seed

        stop = self._stop_sequences()
        if stop:
            options["stop"] = stop

        return options

    def _hf_parameters(self, max_tokens: int, temperature: float) -> dict:
        parameters = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "return_full_text": False,
        }

        top_p = self._config_float("top_p", 1.0)
        if 0.0 < top_p <= 1.0:
            parameters["top_p"] = top_p

        top_k = self._config_int("top_k", 0)
        if top_k > 0:
            parameters["top_k"] = top_k

        repeat_penalty = self._config_float("repeat_penalty", 1.0)
        if repeat_penalty != 1.0:
            parameters["repetition_penalty"] = repeat_penalty

        seed = self._config_int("seed", -1)
        if seed >= 0:
            parameters["seed"] = seed

        stop = self._stop_sequences()
        if stop:
            parameters["stop"] = stop

        return parameters

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
            "Cross-check at least two distinct source URLs before stating factual claims. "
            "For time-sensitive numbers (prices, rates, dates), do not guess. "
            "Do not say 'as of my last update' or imply offline memory limitations. "
            "Ground your answer in the provided web results only. "
            "If internet_search indicates fallback_only=true, or there are no numeric signals/snippets, "
            "you must NOT output a specific numeric value. "
            "Instead, say verification is limited and provide source links only. "
            "If sources conflict or evidence is weak, explicitly say so and report uncertainty. "
            "Provide the final answer now in plain text. "
            "Do not output <tool_call> tags or request more tools."
        )

    @staticmethod
    def _is_tool_protocol_fallback_error(exc: Exception) -> bool:
        """Return True when the backend likely rejected tool-calling fields."""
        if not isinstance(exc, requests.RequestException):
            return False

        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if status in {400, 404, 405, 415, 422, 500, 501, 502, 503}:
            return True

        # Some proxies/backends surface parser issues without stable status mapping.
        text = str(exc).lower()
        if response is not None:
            try:
                text = f"{text} {response.text}".lower()
            except Exception:
                pass

        hints = (
            "tool",
            "tools",
            "tool_choice",
            "unsupported",
            "unknown field",
            "unrecognized field",
            "bad request",
            "invalid request",
        )
        return any(token in text for token in hints)

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
        def sanitize_query(raw_text: str) -> str:
            text = str(raw_text or "").strip()
            if not text:
                return ""

            # Keep the natural-language user request and drop appended attachment payloads.
            attachment_marker = "\n\n--- File:"
            marker_idx = text.find(attachment_marker)
            if marker_idx >= 0:
                text = text[:marker_idx].strip()

            cleaned_lines = []
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("[Image attached:"):
                    continue
                cleaned_lines.append(line)

            text = re.sub(r"\s+", " ", "\n".join(cleaned_lines)).strip()
            return text[:400]

        for msg in reversed(chat_messages or []):
            if str(msg.get("role", "")).lower() != "user":
                continue
            text = self._extract_text_content(msg.get("content", ""))
            if text:
                sanitized = sanitize_query(text)
                if sanitized:
                    return sanitized

        return sanitize_query(str(prompt or "").strip())

    @staticmethod
    def _query_variants(query: str) -> list:
        base = str(query or "").strip()
        if not base:
            return []

        variants = [base]

        # Minimal typo corrections for frequently misspelled high-salience entities.
        typo_map = {
            "artimis": "artemis",
        }

        def apply_typo_map(text: str) -> str:
            tokens = re.split(r"(\W+)", text)
            out = []
            for token in tokens:
                key = token.lower()
                if key in typo_map:
                    replacement = typo_map[key]
                    if token.isupper():
                        replacement = replacement.upper()
                    elif token.istitle():
                        replacement = replacement.title()
                    out.append(replacement)
                else:
                    out.append(token)
            return "".join(out).strip()

        corrected = apply_typo_map(base)
        if corrected and corrected != base:
            variants.append(corrected)

        normalized = re.sub(r"[^\w\s:/.-]", " ", base)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if normalized and normalized not in variants:
            variants.append(normalized)

        # Date-focused rewrite for "when" queries to increase retrieval precision.
        when_match = re.match(r"^\s*when\s+(?:was|is|did|does)?\s*(.+)$", base, flags=re.IGNORECASE)
        if when_match:
            subject = when_match.group(1).strip(" ?.")
            if subject:
                variants.append(f"{subject} date")
                variants.append(f"latest {subject} date")
                variants.append(f"{subject} official date")

        # Generic freshness/source variants.
        variants.append(f"{base} latest")
        variants.append(f"{base} official source")
        if corrected and corrected != base:
            variants.append(f"{corrected} latest")
            variants.append(f"{corrected} official source")

        deduped = []
        seen = set()
        for item in variants:
            cleaned = re.sub(r"\s+", " ", str(item or "")).strip()
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)

        return deduped[:7]

    @staticmethod
    def _should_force_web_search(query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False

        # Skip obvious small-talk turns when internet mode is enabled.
        if text in {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "cool"}:
            return False

        triggers = (
            "internet", "web", "search", "browse", "look up", "lookup",
            "latest", "current", "today", "verify", "fact-check", "fact check",
            "check your results", "news",
        )
        if any(token in text for token in triggers):
            return True

        # Internet mode should generally run a lookup for substantive user queries
        # even without explicit trigger words.
        if "?" in text:
            return True
        if len(text.split()) >= 3:
            return True
        return False

    @staticmethod
    def _is_google_query_link(url: str) -> bool:
        parsed = urlparse(str(url or "").strip())
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if host != "google.com" or parsed.path != "/search":
            return False
        params = parse_qs(parsed.query, keep_blank_values=False)
        return bool(params.get("q"))

    @classmethod
    def _is_low_confidence_web_sources(cls, web_sources: list) -> bool:
        urls = [
            str(item.get("url", "")).strip()
            for item in (web_sources or [])
            if isinstance(item, dict)
        ]
        urls = [u for u in urls if u]
        if not urls:
            return False
        return all(cls._is_google_query_link(url) for url in urls)

    @staticmethod
    def _is_time_sensitive_numeric_query(query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text:
            return False
        numeric_markers = (
            "price", "quote", "rate", "value", "cost", "how much", "trading at",
            "per ounce", "xau", "gold", "silver", "platinum", "oil",
            "btc", "bitcoin", "eth", "ethereum", "stock", "index",
            "current", "latest", "today", "now", "usd", "$",
        )
        return any(marker in text for marker in numeric_markers)

    @staticmethod
    def _build_limited_verification_response(query: str, web_sources: list) -> str:
        lines = [
            "I couldn't verify a reliable live numeric value from the currently retrieved sources.",
            "The available links are search pages (not directly parsed quote pages), so I won't guess a number.",
            "Please open at least two of these sources and confirm the latest timestamped quote:",
        ]
        for idx, source in enumerate((web_sources or [])[:3], start=1):
            if not isinstance(source, dict):
                continue
            title = str(source.get("title", "")).strip() or f"Source {idx}"
            url = str(source.get("url", "")).strip()
            if not url:
                continue
            lines.append(f"{idx}. {title}")
            lines.append(f"   {url}")
        if query:
            lines.append(f"Query reviewed: {query}")
        return "\n".join(lines)

    def _build_web_context_message(self, tool_result: dict) -> str:
        query = str((tool_result or {}).get("query", "")).strip()
        fetched_at = str((tool_result or {}).get("fetched_at", "")).strip()
        results = (tool_result or {}).get("results", []) or []
        signals = (tool_result or {}).get("signals", []) or []
        fallback_only = bool((tool_result or {}).get("fallback_only"))
        error = str((tool_result or {}).get("error", "")).strip()
        lines = [
            "INTERNET_SEARCH_RESULTS (auto-fetched by app):",
            f"Query: {query}" if query else "Query: (empty)",
        ]
        if fetched_at:
            lines.append(f"Fetched at: {fetched_at}")
        if fallback_only:
            lines.append("Verification confidence: LOW (fallback search links only).")
        if results:
            lines.append("Top results:")
            for idx, entry in enumerate(results[:4], start=1):
                title = str(entry.get("title", "")).strip() or "(untitled)"
                url = str(entry.get("url", "")).strip() or "(no url)"
                snippet = str(entry.get("snippet", "")).strip()
                lines.append(f"{idx}. {title}")
                lines.append(f"   URL: {url}")
                if snippet:
                    lines.append(f"   Snippet: {snippet[:140]}")
        if signals:
            lines.append("Numeric signals extracted from snippets:")
            for idx, signal in enumerate(signals[:4], start=1):
                value = str(signal.get("value", "")).strip() or "(unknown)"
                url = str(signal.get("url", "")).strip() or "(no url)"
                context = str(signal.get("context", "")).strip()
                lines.append(f"{idx}. Value: {value}")
                lines.append(f"   URL: {url}")
                if context:
                    lines.append(f"   Context: {context[:120]}")
        elif error:
            lines.append(f"Search error: {error}")
            if "fallback query links" in error.lower():
                lines.append(
                    "IMPORTANT: No parseable source values were extracted. "
                    "Do not output a precise numeric claim."
                )
        if fallback_only and not signals:
            lines.append(
                "MANDATORY: Do not provide a specific number in your answer for this query. "
                "State that verification is limited and cite links."
            )

        lines.append(
            "Use these web results in your next answer. "
            "Cite at least two distinct source URLs for time-sensitive claims when available. "
            "If fewer than two usable sources exist, explicitly state that verification is limited."
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
        has_results = bool((tool_result or {}).get("results"))
        has_error = bool(str((tool_result or {}).get("error", "")).strip())
        if not has_results and not has_error:
            return chat_messages, web_results_used, list(web_sources or [])

        context_msg = self._build_web_context_message(tool_result)
        updated_messages = list(chat_messages)
        updated_messages.append({"role": "system", "content": context_msg})
        merged_sources = self._merge_web_sources(web_sources or [], tool_result, limit=5)

        if has_results:
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

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        results = []
        errors = []
        seen_urls = set()
        fallback_only = False
        google_challenge_detected = False

        def canonical_host(url: str) -> str:
            parsed_url = urlparse(str(url or ""))
            host = parsed_url.netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            return host

        def is_blocked_url(url: str) -> bool:
            parsed_url = urlparse(str(url or ""))
            host = canonical_host(url)
            if parsed_url.scheme not in {"http", "https"}:
                return True
            if not host:
                return True
            blocked_hosts = {
                "google.com",
                "webcache.googleusercontent.com",
                "gstatic.com",
                "accounts.google.com",
                "support.google.com",
                "policies.google.com",
                "maps.google.com",
                "youtube.com",
                "youtu.be",
            }
            return host in blocked_hosts or host.endswith(".google.com")

        def add_result(title: str, url: str, snippet: str, allow_google: bool = False):
            if len(results) >= max_results:
                return
            clean_url = str(url or "").strip()
            if not clean_url or clean_url in seen_urls:
                return
            if is_blocked_url(clean_url) and not allow_google:
                return
            seen_urls.add(clean_url)
            results.append({
                "title": str(title or "").strip()[:180] or clean_url,
                "url": clean_url,
                "snippet": str(snippet or "").strip()[:500]
            })

        def looks_like_google_challenge(html: str) -> bool:
            text = str(html or "").lower()
            if not text:
                return False
            markers = (
                "enablejs",
                "window.sgs",
                "sg_ss",
                "sorry/index",
                "unusual traffic from your computer network",
                "recaptcha",
                "captcha",
            )
            return any(marker in text for marker in markers)

        def clean_html_text(raw_html: str) -> str:
            text = re.sub(r"<[^>]+>", "", str(raw_html or ""))
            text = unescape(text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        def diversify_results(entries: list, limit: int) -> list:
            unique_host_results = []
            overflow = []
            seen_hosts = set()

            for entry in entries:
                if len(unique_host_results) >= limit:
                    break
                host = canonical_host(entry.get("url", ""))
                if host and host not in seen_hosts:
                    seen_hosts.add(host)
                    unique_host_results.append(entry)
                else:
                    overflow.append(entry)

            for entry in overflow:
                if len(unique_host_results) >= limit:
                    break
                unique_host_results.append(entry)

            return unique_host_results[:limit]

        def extract_numeric_signals(entries: list) -> list:
            # Pull candidate price/quantity tokens from snippets to reduce blind guessing.
            pattern = re.compile(
                r"(?:USD|US\$|\$)\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:USD|dollars?)",
                re.IGNORECASE,
            )
            signals = []
            seen = set()
            for entry in entries:
                source_url = str(entry.get("url", "")).strip()
                text = f"{entry.get('title', '')} {entry.get('snippet', '')}"
                for match in pattern.finditer(text):
                    value = match.group(0).strip()
                    key = (value.lower(), source_url)
                    if key in seen:
                        continue
                    seen.add(key)
                    window_start = max(0, match.start() - 30)
                    window_end = min(len(text), match.end() + 30)
                    context = re.sub(r"\s+", " ", text[window_start:window_end]).strip()
                    signals.append({"value": value, "url": source_url, "context": context})
                    if len(signals) >= 8:
                        return signals
            return signals

        def parse_google_html(html: str):
            if not html:
                return

            proxy_link_patterns = [
                re.compile(r'href="/url\?([^"]+)"'),
                re.compile(r"href='/url\?([^']+)'"),
            ]
            direct_link_patterns = [
                re.compile(r'href="(https?://[^"#]+)"'),
                re.compile(r"href='(https?://[^'#]+)'"),
            ]
            title_patterns = [
                re.compile(r"<h3[^>]*>(.*?)</h3>", re.IGNORECASE | re.DOTALL),
                re.compile(r'aria-label="([^"]+)"', re.IGNORECASE | re.DOTALL),
                re.compile(r">([^<]{12,140})</a>", re.IGNORECASE | re.DOTALL),
            ]
            snippet_patterns = [
                re.compile(
                    r'<div[^>]+class="[^"]*(?:VwiC3b|yXK7lf|s3v9rd)[^"]*"[^>]*>(.*?)</div>',
                    re.IGNORECASE | re.DOTALL,
                ),
                re.compile(
                    r'<div[^>]+class="[^"]*(?:ITZIwc|BNeawe s3v9rd AP7Wnd)[^"]*"[^>]*>(.*?)</div>',
                    re.IGNORECASE | re.DOTALL,
                ),
                re.compile(r"<span[^>]*>(.*?)</span>", re.IGNORECASE | re.DOTALL),
            ]

            def parse_match(match, is_proxy: bool):
                if len(results) >= max_results:
                    return

                if is_proxy:
                    query_string = unescape(match.group(1)).replace("&amp;", "&")
                    parsed_query = parse_qs(query_string, keep_blank_values=False)
                    raw_url = (parsed_query.get("q") or parsed_query.get("url") or [None])[0]
                    if not raw_url:
                        return
                    url = unquote(raw_url).strip()
                else:
                    url = unescape(match.group(1)).strip()

                if is_blocked_url(url):
                    return

                start = max(0, match.start() - 450)
                end = min(len(html), match.end() + 3800)
                segment = html[start:end]

                title = ""
                for title_pattern in title_patterns:
                    title_match = title_pattern.search(segment)
                    if not title_match:
                        continue
                    title = clean_html_text(title_match.group(1))
                    if title:
                        break
                if not title:
                    parsed = urlparse(url)
                    title = f"{canonical_host(url)} {parsed.path[:40]}".strip()

                snippet = ""
                for pattern in snippet_patterns:
                    snippet_match = pattern.search(segment)
                    if not snippet_match:
                        continue
                    snippet = clean_html_text(snippet_match.group(1))
                    if snippet:
                        break

                add_result(title, url, snippet)

            for pattern in proxy_link_patterns:
                for match in pattern.finditer(html):
                    parse_match(match, is_proxy=True)
                    if len(results) >= max_results:
                        return

            for pattern in direct_link_patterns:
                for match in pattern.finditer(html):
                    parse_match(match, is_proxy=False)
                    if len(results) >= max_results:
                        return

        def resolve_google_news_item_url(item_url: str, source_url: str = "") -> str:
            candidate = str(item_url or "").strip()
            source_candidate = str(source_url or "").strip()

            if candidate and not is_blocked_url(candidate):
                return candidate

            if candidate:
                try:
                    resolved = requests.get(
                        candidate,
                        headers=headers,
                        allow_redirects=True,
                        timeout=min(8, self.inference_timeout),
                    )
                    resolved.raise_for_status()
                    final_url = str(resolved.url or "").strip()
                    if final_url and not is_blocked_url(final_url):
                        return final_url
                except Exception as exc:
                    errors.append(f"Google News link resolve failed: {exc}")

            if source_candidate and not is_blocked_url(source_candidate):
                return source_candidate
            return candidate

        def parse_google_news_rss(xml_text: str):
            if not xml_text:
                return
            try:
                root = ET.fromstring(xml_text)
            except ET.ParseError as exc:
                errors.append(f"Google News RSS parse failed: {exc}")
                return

            for item in root.findall("./channel/item"):
                if len(results) >= max_results:
                    break

                title = str(item.findtext("title", "")).strip()
                link = str(item.findtext("link", "")).strip()
                description = clean_html_text(item.findtext("description", ""))
                pub_date = str(item.findtext("pubDate", "")).strip()
                source_elem = item.find("source")
                source_name = ""
                source_url = ""
                if source_elem is not None:
                    source_name = str(source_elem.text or "").strip()
                    source_url = str(source_elem.get("url") or "").strip()

                resolved_url = resolve_google_news_item_url(link, source_url=source_url)
                if not resolved_url:
                    continue

                if source_name and source_name.lower() not in title.lower():
                    result_title = f"{title} ({source_name})" if title else source_name
                else:
                    result_title = title

                snippet_parts = []
                if description:
                    snippet_parts.append(description)
                if pub_date:
                    snippet_parts.append(f"Published: {pub_date}")
                snippet = " | ".join(snippet_parts)

                add_result(result_title or resolved_url, resolved_url, snippet)

        query_variants = self._query_variants(query)

        attempt_modes = [
            {"safe": "off"},
            {"gbv": "1"},  # Basic HTML fallback
        ]

        for variant in query_variants:
            if len(results) >= max_results:
                break
            for mode in attempt_modes:
                if len(results) >= max_results:
                    break
                params = {
                    "q": variant,
                    "num": max(max_results, 6),
                    "hl": "en",
                    "gl": "us",
                    "pws": "0",
                    **mode,
                }
                try:
                    response = requests.get(
                        "https://www.google.com/search",
                        params=params,
                        headers=headers,
                        timeout=min(9, self.inference_timeout),
                    )
                    response.raise_for_status()
                    html = response.text
                    if looks_like_google_challenge(html):
                        google_challenge_detected = True
                        errors.append(
                            "Google search HTML returned anti-bot/challenge page; "
                            "switching to Google RSS fallback."
                        )
                        continue
                    parse_google_html(html)
                except Exception as exc:
                    errors.append(f"Google ({variant[:80]}): {exc}")

        if len(results) < max_results:
            for variant in query_variants:
                if len(results) >= max_results:
                    break
                try:
                    response = requests.get(
                        "https://news.google.com/rss/search",
                        params={
                            "q": variant,
                            "hl": "en-US",
                            "gl": "US",
                            "ceid": "US:en",
                        },
                        headers=headers,
                        timeout=min(9, self.inference_timeout),
                    )
                    response.raise_for_status()
                    before = len(results)
                    parse_google_news_rss(response.text)
                    if len(results) == before:
                        errors.append(
                            f"Google News RSS ({variant[:80]}): no parseable items."
                        )
                except Exception as exc:
                    errors.append(f"Google News RSS ({variant[:80]}): {exc}")

        if not results and not errors:
            errors.append(
                "Google: no parseable search results returned "
                "(possibly blocked or response format changed)."
            )

        real_results_count = len(results)
        if not results:
            fallback_only = True
            fallback_queries = [
                query,
                f"{query} site:reuters.com",
                f"{query} site:bloomberg.com",
                f"{query} site:marketwatch.com",
            ]
            for q in fallback_queries:
                if len(results) >= max_results:
                    break
                add_result(
                    f"Google search results for: {q[:120]}",
                    f"https://www.google.com/search?q={quote_plus(q)}",
                    "Open this Google results page and verify using multiple publisher sources.",
                    allow_google=True,
                )

        diversified = diversify_results(results, limit=max_results)
        signals = extract_numeric_signals(diversified if real_results_count > 0 else [])

        result_payload = {
            "query": query,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "results": diversified[:max_results],
            "signals": signals[:8],
            "fallback_only": bool(fallback_only),
        }
        if errors and real_results_count == 0 and fallback_only:
            result_payload["warnings"] = list(dict.fromkeys(errors))[:4]
            if google_challenge_detected:
                result_payload["error"] = (
                    "Google search was blocked by an anti-bot/challenge response and "
                    "Google News RSS did not provide enough parseable publisher links; "
                    "using fallback query links. Open at least two sources before trusting numeric claims."
                )
            else:
                result_payload["error"] = (
                    "Google returned no parseable organic results; using fallback query links. "
                    "Open at least two sources before trusting numeric claims."
                )
        elif errors and real_results_count == 0:
            result_payload["error"] = " | ".join(list(dict.fromkeys(errors))[:4])
        elif errors:
            result_payload["warnings"] = list(dict.fromkeys(errors))[:4]
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

        if getattr(self, "llama_server_tool_protocol_supported", None) is False:
            return resolved_messages, web_results_used, web_sources

        for _ in range(max_rounds):
            try:
                payload = {
                    "messages": resolved_messages,
                    "stream": False,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "tools": tool_spec,
                    "tool_choice": "auto",
                }
                payload.update(self._llama_server_request_extras())
                payload = self._deep_merge_dicts(payload, self._api_extra_body())
                response = requests.post(
                    self._llama_server_chat_url(self.llama_server_url),
                    headers=self._llama_server_headers(self.llama_server_api_key),
                    json=payload,
                    timeout=self.inference_timeout,
                )
                response.raise_for_status()
                self.llama_server_tool_protocol_supported = True
            except requests.RequestException as exc:
                if self._is_tool_protocol_fallback_error(exc):
                    self.llama_server_tool_protocol_supported = False
                    print(
                        "⚠️  llama-server rejected tool-calling payload; "
                        "falling back to non-tool generation."
                    )
                    break
                raise
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
                    web_sources = self._merge_web_sources(web_sources, tool_result, limit=5)
                    if (tool_result.get("results") or []):
                        web_results_used += 1
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

    def _resolve_openai_compatible_internet_tools(
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

        if getattr(self, "openai_tool_protocol_supported", None) is False:
            return resolved_messages, web_results_used, web_sources

        for _ in range(max_rounds):
            try:
                payload = {
                    "model": request_model,
                    "messages": resolved_messages,
                    "stream": False,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "tools": tool_spec,
                    "tool_choice": "auto",
                }
                payload.update(self._openai_request_extras())
                payload = self._deep_merge_dicts(payload, self._api_extra_body())
                response = requests.post(
                    self._openai_compatible_chat_url(base_url),
                    headers=self._openai_compatible_headers(api_key),
                    json=payload,
                    timeout=self.inference_timeout,
                )
                response.raise_for_status()
                self.openai_tool_protocol_supported = True
            except requests.RequestException as exc:
                if self._is_tool_protocol_fallback_error(exc):
                    self.openai_tool_protocol_supported = False
                    print(
                        "⚠️  OpenAI-compatible endpoint rejected tool-calling payload; "
                        "falling back to non-tool generation."
                    )
                    break
                raise

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
                    web_sources = self._merge_web_sources(web_sources, tool_result, limit=5)
                    if (tool_result.get("results") or []):
                        web_results_used += 1
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
            payload = {
                "model": target['request_model'],
                "messages": resolved_messages,
                "stream": False,
                "tools": tool_spec,
                "options": self._ollama_options(max_tokens, temperature),
            }
            response = requests.post(
                self._ollama_api_url(target['base_url'], 'chat'),
                headers=target['headers'],
                json=payload,
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
                    web_sources = self._merge_web_sources(web_sources, tool_result, limit=5)
                    if (tool_result.get("results") or []):
                        web_results_used += 1
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
        elif self.backend_type == BackendType.OPENAI_COMPATIBLE:
            yield from self._openai_compatible_generate(
                model, prompt, max_tokens, temperature, callback, messages, internet_enabled
            )
    
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
        latest_query = self._latest_user_query(chat_messages, prompt=prompt)
        if (
            internet_enabled
            and forced_web_results_used > 0
            and self._is_time_sensitive_numeric_query(latest_query)
            and self._is_low_confidence_web_sources(forced_web_sources)
        ):
            guarded_text = self._build_limited_verification_response(latest_query, forced_web_sources)
            if callback:
                callback(guarded_text)
            yield guarded_text
            if hasattr(self.local_wrapper, "last_generation_stats"):
                self.local_wrapper.last_generation_stats = {
                    "prompt_tps": None,
                    "generation_tps": None,
                    "prompt_tokens": max(1, sum(self._message_content_length(m.get("content", "")) for m in chat_messages) // 4),
                    "completion_tokens": max(1, len(guarded_text) // 4),
                    "web_results_used": forced_web_results_used,
                    "web_sources": forced_web_sources[:5],
                }
            return

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
            latest_query = self._latest_user_query(chat_messages, prompt=prompt)
            if (
                internet_enabled
                and web_results_used > 0
                and self._is_time_sensitive_numeric_query(latest_query)
                and self._is_low_confidence_web_sources(web_sources)
            ):
                guarded_text = self._build_limited_verification_response(latest_query, web_sources)
                if callback:
                    callback(guarded_text)
                yield guarded_text
                self.last_generation_stats = {
                    "prompt_tps": None,
                    "generation_tps": None,
                    "prompt_tokens": max(1, sum(self._message_content_length(m.get("content", "")) for m in chat_messages) // 4),
                    "completion_tokens": max(1, len(guarded_text) // 4),
                    "web_results_used": web_results_used,
                    "web_sources": web_sources[:5],
                }
                return
            payload = {
                "messages": chat_messages,
                "stream": True,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream_options": {"include_usage": True},
            }
            payload.update(self._llama_server_request_extras())
            payload = self._deep_merge_dicts(payload, self._api_extra_body())

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
            latest_query = self._latest_user_query(chat_messages, prompt=prompt)
            if (
                internet_enabled
                and web_results_used > 0
                and self._is_time_sensitive_numeric_query(latest_query)
                and self._is_low_confidence_web_sources(web_sources)
            ):
                guarded_text = self._build_limited_verification_response(latest_query, web_sources)
                if callback:
                    callback(guarded_text)
                yield guarded_text
                self.last_generation_stats = {
                    "prompt_tps": None,
                    "generation_tps": None,
                    "prompt_tokens": max(1, sum(self._message_content_length(m.get("content", "")) for m in chat_messages) // 4),
                    "completion_tokens": max(1, len(guarded_text) // 4),
                    "web_results_used": web_results_used,
                    "web_sources": web_sources[:5],
                }
                return
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
            response_text = e.response.text if e.response is not None else ""
            if self._is_ollama_chat_template_error(status_code, response_text):
                try:
                    prompt = self._ollama_prompt_from_messages(chat_messages)
                    response = requests.post(
                        self._ollama_api_url(target['base_url'], 'generate'),
                        headers=target['headers'],
                        json={
                            "model": target['request_model'],
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "num_predict": max_tokens,
                                "temperature": temperature,
                            },
                        },
                        timeout=self.inference_timeout,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    text = payload.get('response', '') or ''
                    if callback:
                        callback(text)
                    yield text
                    self.last_generation_stats = {
                        "prompt_tps": None,
                        "generation_tps": None,
                        "prompt_tokens": max(1, len(prompt) // 4),
                        "completion_tokens": max(1, len(text) // 4),
                        "web_results_used": 0,
                        "web_sources": [],
                    }
                    return
                except Exception as fallback_error:
                    raise RuntimeError(f"Ollama API error: {fallback_error}") from fallback_error
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

    def _openai_compatible_generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        callback: Optional[Callable[[str], None]],
        messages: list = None,
        internet_enabled: bool = False,
    ) -> Generator[str, None, None]:
        """Generate using an OpenAI-compatible /v1/chat/completions endpoint."""
        try:
            request_start = time.time()
            first_token_time = None
            full_response = ""
            stream_usage = None
            web_results_used = 0
            web_sources = []

            # Use model data from the selected combo item if it's a dict,
            # otherwise fall back to the default config.
            if isinstance(model, dict):
                base_url = model.get("base_url", self.openai_base_url)
                api_key = model.get("api_key", self.openai_api_key)
                request_model = model.get("model", self.openai_model)
            else:
                base_url = self.openai_base_url
                api_key = self.openai_api_key
                request_model = str(model or self.openai_model).strip() or self.openai_model

            chat_messages = messages if messages else [{"role": "user", "content": prompt}]
            if internet_enabled:
                chat_messages, web_results_used, web_sources = self._resolve_openai_compatible_internet_tools(
                    chat_messages, max_tokens=max_tokens, temperature=temperature, base_url=base_url, api_key=api_key, request_model=request_model
                )
            chat_messages, web_results_used, web_sources = self._apply_forced_web_context_if_needed(
                chat_messages,
                prompt=prompt,
                internet_enabled=internet_enabled,
                web_results_used=web_results_used,
                web_sources=web_sources,
            )
            latest_query = self._latest_user_query(chat_messages, prompt=prompt)
            if (
                internet_enabled
                and web_results_used > 0
                and self._is_time_sensitive_numeric_query(latest_query)
                and self._is_low_confidence_web_sources(web_sources)
            ):
                guarded_text = self._build_limited_verification_response(latest_query, web_sources)
                if callback:
                    callback(guarded_text)
                yield guarded_text
                self.last_generation_stats = {
                    "prompt_tps": None,
                    "generation_tps": None,
                    "prompt_tokens": max(1, sum(self._message_content_length(m.get("content", "")) for m in chat_messages) // 4),
                    "completion_tokens": max(1, len(guarded_text) // 4),
                    "web_results_used": web_results_used,
                    "web_sources": web_sources[:5],
                }
                return

            response = requests.post(
                self._openai_compatible_chat_url(base_url),
                headers=self._openai_compatible_headers(api_key),
                json={
                    "model": request_model,
                    "messages": chat_messages,
                    "stream": True,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
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

                data_str = line_str[6:].strip()
                if data_str == '[DONE]':
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
                if isinstance(content, list):
                    content = ''.join(
                        part.get('text', '') if isinstance(part, dict) else str(part)
                        for part in content
                    )
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
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code == 401:
                raise RuntimeError("OpenAI-compatible API error: 401 Unauthorized.")
            raise RuntimeError(f"OpenAI-compatible API error: {e}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenAI-compatible API error: {e}")
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
        if self.backend_type in {BackendType.LLAMA_SERVER, BackendType.OLLAMA, BackendType.OPENAI_COMPATIBLE}:
            return dict(self.last_generation_stats)
        return {}
    
    def stop_generation(self):
        """Stop current generation"""
        if self.backend_type == BackendType.LOCAL:
            self.local_wrapper.stop_generation()
        elif self.backend_type in {BackendType.LLAMA_SERVER, BackendType.OLLAMA, BackendType.OPENAI_COMPATIBLE}:
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
    def test_openai_compatible_connection(base_url: str = 'https://api.openai.com/v1', api_key: str = '') -> bool:
        """Test if an OpenAI-compatible endpoint is reachable."""
        try:
            response = requests.get(
                f"{UnifiedBackend._openai_compatible_base_url(base_url)}/models",
                headers=UnifiedBackend._openai_compatible_headers(api_key),
                timeout=5,
            )
            return response.status_code in {200, 401, 403, 404}
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
