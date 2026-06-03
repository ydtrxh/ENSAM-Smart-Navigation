"""
LLM backend abstraction for the ENSAM navigation NLP module.

The module exposes one interface for Ollama during development and
llama-cpp-python for local GGUF inference. Model-specific prompt handling is
kept here: Qwen-style models may use /no_think, while Llama 3.2 receives the
system prompt as-is.
"""

from __future__ import annotations

import abc
import json
import logging

import requests

logger = logging.getLogger(__name__)


class LLMBackendError(RuntimeError):
    """Base exception for LLM backend failures."""


class OllamaConnectionError(LLMBackendError):
    """Raised when the Ollama API is unreachable."""


class LLMResponseError(LLMBackendError):
    """Raised when a backend returns an unexpected response payload."""


class LLMBackend(abc.ABC):
    """Abstract base for all LLM backends used by the NLP pipeline."""

    @abc.abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        """Run inference and return the model's raw text response."""

    @staticmethod
    def _needs_no_think(model_name: str) -> bool:
        return "qwen" in model_name.lower()

    @classmethod
    def _prepare_system_prompt(cls, system_prompt: str, model_name: str) -> str:
        """
        Add /no_think only for models that require it.

        Llama 3.2 does not use /no_think, so the system prompt is passed as-is.
        """
        if cls._needs_no_think(model_name):
            return f"/no_think\n{system_prompt}"
        return system_prompt


class OllamaBackend(LLMBackend):
    """Ollama chat backend using the local HTTP API."""

    OLLAMA_URL = "http://localhost:11434/api/chat"
    OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

    def __init__(self, model: str = "llama3.2:latest"):
        self.model = model
        logger.info("OllamaBackend initialized with model=%s", self.model)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        full_system = self._prepare_system_prompt(system_prompt, self.model)
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 256},
            "messages": [
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            response = requests.post(self.OLLAMA_URL, json=payload, timeout=60)
            if response.status_code == 404:
                logger.warning("Ollama /api/chat returned 404; retrying with /api/generate.")
                return self._generate_completion(full_system, user_prompt, temperature)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                "Ollama is not running. Start it with: ollama serve - "
                f"then run: ollama pull {self.model}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaConnectionError("Ollama request timed out after 60s.") from exc
        except requests.exceptions.HTTPError as exc:
            raise LLMResponseError(f"Ollama HTTP error: {exc}") from exc
        except (KeyError, json.JSONDecodeError) as exc:
            raise LLMResponseError(f"Unexpected Ollama response format: {exc}") from exc

    def _generate_completion(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        prompt = f"{system_prompt}\n\nUser message:\n{user_prompt}\n\nJSON:"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 256},
        }
        try:
            response = requests.post(self.OLLAMA_GENERATE_URL, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["response"]
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                "Ollama is not running. Start it with: ollama serve - "
                f"then run: ollama pull {self.model}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaConnectionError("Ollama generate request timed out after 60s.") from exc
        except requests.exceptions.HTTPError as exc:
            raise LLMResponseError(f"Ollama generate HTTP error: {exc}") from exc
        except (KeyError, json.JSONDecodeError) as exc:
            raise LLMResponseError(f"Unexpected Ollama generate response format: {exc}") from exc


class LlamaCppBackend(LLMBackend):
    """llama-cpp-python backend for local GGUF models."""

    def __init__(self, model_path: str, n_threads: int = 4):
        self.model_path = model_path
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise LLMBackendError(
                "llama-cpp-python is not installed. Install it with: pip install llama-cpp-python"
            ) from exc

        logger.info("LlamaCppBackend loading GGUF from: %s", model_path)
        self._llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=n_threads,
            verbose=False,
        )
        logger.info("LlamaCppBackend model loaded.")

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        full_system = self._prepare_system_prompt(system_prompt, self.model_path)
        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=256,
        )
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMResponseError(f"Unexpected llama-cpp response format: {exc}") from exc


def get_backend(config: dict) -> LLMBackend:
    """Create an LLM backend from config.yaml values."""
    backend_name = config.get("backend", "ollama").lower()

    if backend_name == "ollama":
        return OllamaBackend(model=config.get("ollama_model", "llama3.2:latest"))

    if backend_name == "llamacpp":
        return LlamaCppBackend(
            model_path=config.get("llamacpp_model_path", "models/llama-3.2-3b-instruct.Q4_K_M.gguf"),
            n_threads=config.get("llamacpp_n_threads", 4),
        )

    raise ValueError(f"Unknown backend '{backend_name}'. Valid options: 'ollama', 'llamacpp'.")
