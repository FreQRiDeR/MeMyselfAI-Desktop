import unittest

from backend.unified_backend import UnifiedBackend


class OllamaChatTemplateFallbackTests(unittest.TestCase):
    def test_detects_missing_chat_template_error(self):
        self.assertTrue(
            UnifiedBackend._is_ollama_chat_template_error(
                400,
                "model does not support chat: no chat template available",
            )
        )

    def test_builds_plain_prompt_from_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        prompt = UnifiedBackend._ollama_prompt_from_messages(messages)
        self.assertIn("System:", prompt)
        self.assertIn("User: Hello", prompt)
        self.assertIn("Assistant: Hi there", prompt)
        self.assertIn("User: How are you?", prompt)


if __name__ == "__main__":
    unittest.main()
