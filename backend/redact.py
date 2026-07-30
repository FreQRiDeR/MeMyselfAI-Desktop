"""
redact.py
Functions for redacting sensitive information from logs.
"""

import logging
import re
import json
from typing import Any

REDACTED_PLACEHOLDER = "[REDACTED]"

# Global list of secrets to be populated at startup
SENSITIVE_STRINGS = []

# Keywords that indicate a value is sensitive and should be redacted.
# Case-insensitive.
SENSITIVE_KEY_PATTERNS = [
    "key",
    "token",
    "secret",
    "password",
    "auth",
    "bearer",
    "credential",
]


def _is_sensitive_key(key: str) -> bool:
    """Check if a dictionary key suggests its value is sensitive."""
    if not isinstance(key, str):
        return False
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in SENSITIVE_KEY_PATTERNS)


def redact(data: Any) -> Any:
    """
    Recursively redact sensitive information from various data types.

    - Redacts values in dictionaries where the key is sensitive.
    - Redacts 'Authorization' headers in dictionaries.
    - Redacts bearer tokens in strings.
    """
    if isinstance(data, dict):
        redacted_dict = {}
        for k, v in data.items():
            if _is_sensitive_key(k) or (isinstance(k, str) and k.lower() == "authorization"):
                redacted_dict[k] = REDACTED_PLACEHOLDER
            else:
                redacted_dict[k] = redact(v)
        return redacted_dict

    if isinstance(data, list):
        return [redact(item) for item in data]

    if isinstance(data, str):
        # Redact bearer tokens (case-insensitive)
        data = re.sub(r'(bearer\s+)[^\s]+', r'\1' + REDACTED_PLACEHOLDER, data, flags=re.IGNORECASE)
        # Redact API keys in key=value format
        for pattern in SENSITIVE_KEY_PATTERNS:
            data = re.sub(
                f'({pattern}=)[^&\\s\'"]+',
                r'\1' + REDACTED_PLACEHOLDER,
                data,
                flags=re.IGNORECASE
            )
        return data

    return data


def redact_command(command: list) -> list:
    """Redact sensitive arguments from a command list."""
    redacted_cmd = []
    for i, arg in enumerate(command):
        # If the previous argument was a sensitive flag (e.g., --api-key), redact the current one.
        if i > 0 and _is_sensitive_key(command[i-1]):
            redacted_cmd.append(REDACTED_PLACEHOLDER)
        else:
            redacted_cmd.append(redact(arg))
    return redacted_cmd


class RedactingFilter(logging.Filter):
    """
    A logging filter that redacts specific sensitive strings from log messages.
    """
    def __init__(self, name: str = ""):
        super().__init__(name)

    def filter(self, record: logging.LogRecord) -> bool:
        # Redact any known sensitive strings from the log message.
        # This is a broad but effective approach.
        if SENSITIVE_STRINGS:
            for secret in SENSITIVE_STRINGS:
                if secret in record.msg:
                    record.msg = record.msg.replace(secret, REDACTED_PLACEHOLDER)
        return True # Always allow the record to pass through