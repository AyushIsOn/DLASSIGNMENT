"""Text generation backends. All of them receive the fully rendered training prompt.

* TransformersGenerator  - base model + LoRA adapter (or a merged model) on the GPU
* OllamaGenerator        - the exported GGUF model in Ollama (e.g. on a Mac), raw prompt
* EchoGenerator          - deterministic stand-in for tests
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from acharya.prompting import STOP_STRINGS, clean_generation


class GenerationError(RuntimeError):
    """The model backend failed or is unreachable."""


class Generator(Protocol):
    name: str

    def generate(self, prompt: str, max_new_tokens: int) -> str: ...

    def ready(self) -> tuple[bool, str | None]: ...


class EchoGenerator:
    name = "echo"

    def __init__(self, answer: str = "Vata, Pitta and Kapha are the three doshas.") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        self.prompts.append(prompt)
        return self.answer

    def ready(self) -> tuple[bool, str | None]:
        return True, None


class TransformersGenerator:
    """Loads lazily on first use (or eagerly with `load()`); generation is serialized."""

    def __init__(self, model_dir: Path, adapter_dir: Path | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None
        self.name = f"transformers:{(self.adapter_dir or self.model_dir).name}"
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()
        self._error: str | None = None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                cuda = torch.cuda.is_available()
                dtype = torch.bfloat16 if cuda else torch.float32
                source = self.adapter_dir if self.adapter_dir else self.model_dir
                tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_dir, dtype=dtype, local_files_only=True,
                    attn_implementation="sdpa", device_map={"": 0} if cuda else None)
                if self.adapter_dir:
                    from peft import PeftModel

                    model = PeftModel.from_pretrained(model, self.adapter_dir)
                model.eval()
                self._tokenizer, self._model = tokenizer, model
            except Exception as error:  # surfaced through /health/ready
                self._error = f"{type(error).__name__}: {error}"
                raise GenerationError(self._error) from error

    def ready(self) -> tuple[bool, str | None]:
        return self._model is not None, self._error or (None if self._model else "loading")

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        import torch

        self.load()
        tokenizer, model = self._tokenizer, self._model
        stops = [tokenizer.convert_tokens_to_ids(token) for token in STOP_STRINGS]
        with self._lock, torch.inference_mode():
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                    eos_token_id=stops, pad_token_id=tokenizer.pad_token_id)
        new_tokens = output[0, encoded["input_ids"].shape[1]:]
        return clean_generation(tokenizer.decode(new_tokens, skip_special_tokens=False))


class OllamaGenerator:
    """Calls Ollama's /api/generate with raw=true so Ollama's own chat template is bypassed
    and the model sees exactly the training prompt format."""

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434",
                 timeout: float = 180.0, num_ctx: int = 4096) -> None:
        if not model.strip():
            raise ValueError("an Ollama model name is required")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.name = f"ollama:{model}"

    def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return dict(json.loads(response.read().decode()))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
            raise GenerationError(f"Ollama request failed: {error}") from error

    def ready(self) -> tuple[bool, str | None]:
        try:
            self._post("/api/show", {"model": self.model}, timeout=5)
            return True, None
        except GenerationError as error:
            return False, f"{error} (is `ollama serve` running and `{self.model}` created?)"

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        body = self._post("/api/generate", {
            "model": self.model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "top_k": 1, "num_predict": max_new_tokens,
                        "num_ctx": self.num_ctx, "repeat_penalty": 1.0,
                        "stop": list(STOP_STRINGS)},
        }, timeout=self.timeout)
        text = clean_generation(str(body.get("response", "")))
        if not text:
            raise GenerationError("Ollama returned an empty answer")
        return text
