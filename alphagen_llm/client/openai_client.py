from dataclasses import asdict
import time
from openai import OpenAI
import tokentrim as tt
from tokentrim.model_map import MODEL_MAX_TOKENS

from .base import ChatClient, ChatConfig

DEFAULT_MODEL = "gpt-4o-2024-11-20"

class OpenAIClient(ChatClient):
    def __init__(
        self,
        client: OpenAI,
        config: ChatConfig,
        model: str = DEFAULT_MODEL,
        trim_to_token_limit: bool = True,
        temperature: float = 0.0,
        seed: int = 0,
        max_calls: int = 1000,
        max_tokens: int = 2048
    ) -> None:
        super().__init__(config)
        _update_model_max_tokens()
        self._client = client
        self._model = model
        self._trim = trim_to_token_limit
        self.temperature, self.seed = temperature, seed
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.usage = dict(calls=0, prompt_tokens=0, completion_tokens=0, cached_tokens=0, latency_seconds=0.0)
        self.request_usage = []

    @property
    def model(self):
        return self._model

    def chat_complete(self, content: str) -> str:
        if self.usage["calls"] >= self.max_calls:
            raise RuntimeError("LLM call budget exhausted")
        self._add_message("user", content)
        idx = int(self._system_prompt is not None)
        messages = [asdict(msg) for msg in self._dialog[idx:]]
        if self._trim:
            messages = tt.trim(messages, self._model, self._system_prompt)
        elif self._system_prompt:
            messages.insert(0, {"role": "system", "content": self._system_prompt})
        self.usage["calls"] += 1
        started = time.monotonic()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=self.max_tokens,
            store=False,
        )
        self.usage["latency_seconds"] += time.monotonic() - started
        if response.usage is not None:
            self.usage["prompt_tokens"] += response.usage.prompt_tokens
            self.usage["completion_tokens"] += response.usage.completion_tokens
            self.usage["cached_tokens"] += getattr(response.usage.prompt_tokens_details, "cached_tokens", 0) or 0
        result = response.choices[0].message.content
        if result is None:
            raise ValueError("LLM returned no text")
        self._add_message("assistant", result)
        self.request_usage.append(dict(
            call=self.usage["calls"], model=response.model if hasattr(response, "model") else self._model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
            latency_seconds=time.monotonic() - started,
        ))
        return result

    def _on_reset(self) -> None:
        self._start_idx = 0

    _client: OpenAI
    _model: str


_UPDATED = False


def _update_model_max_tokens():
    global _UPDATED
    if _UPDATED:
        return
    MODEL_MAX_TOKENS["gpt-3.5-turbo"] = 16385
    MODEL_MAX_TOKENS["gpt-3.5-turbo-1106"] = 16385
    MODEL_MAX_TOKENS["gpt-3.5-turbo-0125"] = 16385
    MODEL_MAX_TOKENS["gpt-4o-mini-2024-07-18"] = 128000
    MODEL_MAX_TOKENS["gpt-4o-mini"] = 128000
    MODEL_MAX_TOKENS[DEFAULT_MODEL] = 128000
    MODEL_MAX_TOKENS["gpt-4o"] = 128000
    _UPDATED = True
