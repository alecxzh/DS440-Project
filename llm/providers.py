from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, Optional, List


ProviderName = Literal["groq", "gemini", "zai"]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    thoughts: Optional[str] = None


class BaseLLM:
    provider: ProviderName

    def generate(self, prompt: str) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


_ROTATE_STATUS_CODES = {401, 403, 429}
_ROTATE_TEXT_RE = re.compile(
    r"(rate limit|ratelimit|quota|exceed|exceeded|insufficient|unauthorized|forbidden|invalid api key|api key|permission|billing)",
    re.IGNORECASE,
)


def _exc_status_code(exc: Exception) -> Optional[int]:
    """
    Try to extract an HTTP-ish status code from various SDK exception shapes.
    """
    for attr in ("status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val

    resp = getattr(exc, "response", None)
    if resp is not None:
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            return sc

    return None


def _should_rotate_key(exc: Exception) -> bool:
    """
    Only rotate keys when the error is likely key-specific:
    - auth/permission problems
    - quota / rate-limit issues
    """
    sc = _exc_status_code(exc)
    if sc in _ROTATE_STATUS_CODES:
        return True
    msg = str(exc) or repr(exc)
    return bool(_ROTATE_TEXT_RE.search(msg))


def _split_keys(env_val: Optional[str]) -> List[str]:
    if not env_val:
        return []
    raw = env_val.replace("\n", ",")
    return [k.strip() for k in raw.split(",") if k and k.strip()]


# -----------------------------
# Groq
# -----------------------------


@dataclass(frozen=True)
class GroqConfig:
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_output_tokens: int = 700


class GroqLLM(BaseLLM):
    provider: ProviderName = "groq"

    def __init__(self, api_key: Optional[str] = None, config: Optional[GroqConfig] = None):
        self.config = config or GroqConfig()
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Missing GROQ_API_KEY (env) or api_key parameter.")

        from groq import Groq  # lazy import

        self._client = Groq(api_key=key)

    def generate(self, prompt: str) -> LLMResponse:
        chat_completion = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_output_tokens,
        )
        msg = chat_completion.choices[0].message
        return LLMResponse(text=msg.content or "", thoughts=None)


class RotatingGroqLLM(BaseLLM):
    provider: ProviderName = "groq"

    def __init__(self, api_keys: List[str], config: Optional[GroqConfig] = None):
        keys = [k.strip() for k in api_keys if k and k.strip()]
        if not keys:
            raise ValueError("Provide at least one Groq API key.")

        self.api_keys = keys
        self.config = config or GroqConfig()
        self._idx = 0

        from groq import Groq  # lazy import

        self._Groq = Groq
        self._client = self._Groq(api_key=self.api_keys[self._idx])

    def _rotate(self) -> None:
        self._idx = (self._idx + 1) % len(self.api_keys)
        self._client = self._Groq(api_key=self.api_keys[self._idx])

    def generate(self, prompt: str) -> LLMResponse:
        last_exc: Optional[Exception] = None
        for _ in range(len(self.api_keys)):
            try:
                chat_completion = self._client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_completion_tokens=self.config.max_output_tokens,
                )
                msg = chat_completion.choices[0].message
                return LLMResponse(text=msg.content or "", thoughts=None)
            except Exception as e:
                if _should_rotate_key(e):
                    last_exc = e
                    self._rotate()
                    continue
                raise
        raise last_exc or RuntimeError("All Groq API keys failed.")


# -----------------------------
# Gemini
# -----------------------------


@dataclass(frozen=True)
class GeminiConfig:
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.0
    max_output_tokens: int = 900


class GeminiLLM(BaseLLM):
    provider: ProviderName = "gemini"

    def __init__(self, api_key: Optional[str] = None, config: Optional[GeminiConfig] = None):
        self.config = config or GeminiConfig()
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Missing GEMINI_API_KEY (env) or api_key parameter.")

        from google import genai  # lazy import
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=key)

    def generate(self, prompt: str) -> LLMResponse:
        gen_config = self._types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        response = self._client.models.generate_content(
            model=self.config.model_name,
            contents=prompt,
            config=gen_config,
        )
        return LLMResponse(text=getattr(response, "text", "") or "", thoughts=None)


class RotatingGeminiLLM(BaseLLM):
    provider: ProviderName = "gemini"

    def __init__(self, api_keys: List[str], config: Optional[GeminiConfig] = None):
        keys = [k.strip() for k in api_keys if k and k.strip()]
        if not keys:
            raise ValueError("Provide at least one Gemini API key.")

        self.api_keys = keys
        self.config = config or GeminiConfig()
        self._idx = 0

        from google import genai  # lazy import
        from google.genai import types

        self._types = types
        self._genai = genai
        self._client = self._genai.Client(api_key=self.api_keys[self._idx])

    def _rotate(self) -> None:
        self._idx = (self._idx + 1) % len(self.api_keys)
        self._client = self._genai.Client(api_key=self.api_keys[self._idx])

    def generate(self, prompt: str) -> LLMResponse:
        last_exc: Optional[Exception] = None
        gen_config = self._types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        for _ in range(len(self.api_keys)):
            try:
                response = self._client.models.generate_content(
                    model=self.config.model_name,
                    contents=prompt,
                    config=gen_config,
                )
                return LLMResponse(text=getattr(response, "text", "") or "", thoughts=None)
            except Exception as e:
                if _should_rotate_key(e):
                    last_exc = e
                    self._rotate()
                    continue
                raise
        raise last_exc or RuntimeError("All Gemini API keys failed.")


# -----------------------------
# Z.ai (OpenAI-compatible API)
# See: https://docs.z.ai/guides/develop/openai/python
# -----------------------------


@dataclass(frozen=True)
class ZaiConfig:
    model_name: str = "glm-4.7-flash"
    temperature: float = 0.0
    # GLM-4.7 may spend completion budget on reasoning unless thinking is disabled; keep headroom.
    max_output_tokens: int = 4096
    base_url: str = "https://api.z.ai/api/paas/v4/"


def _zai_max_output_tokens(config: ZaiConfig) -> int:
    raw = os.getenv("ZAI_MAX_OUTPUT_TOKENS") or os.getenv("ZAI_MAX_TOKENS")
    if raw and raw.strip():
        return int(raw.strip())
    return int(config.max_output_tokens)


def _zai_extra_body() -> dict:
    """
    GLM-4.7/5 enable "thinking" by default; reasoning can consume max_tokens and leave content empty.
    See https://docs.z.ai/guides/capabilities/thinking-mode
    """
    v = (os.getenv("ZAI_THINKING") or "disabled").strip().lower()
    if v in ("1", "true", "yes", "enabled", "on"):
        return {"thinking": {"type": "enabled"}}
    return {"thinking": {"type": "disabled"}}


class ZaiLLM(BaseLLM):
    """GLM via Z.ai OpenAI-compatible endpoint."""

    provider: ProviderName = "zai"

    def __init__(self, api_key: Optional[str] = None, config: Optional[ZaiConfig] = None):
        self.config = config or ZaiConfig()
        key = api_key or os.getenv("ZAI_API_KEY")
        if not key:
            raise ValueError("Missing ZAI_API_KEY (env) or api_key parameter.")

        base = os.getenv("ZAI_BASE_URL") or self.config.base_url
        from openai import OpenAI  # lazy import

        self._client = OpenAI(api_key=key, base_url=base)

    @staticmethod
    def _effective_temperature(t: float) -> float:
        # Z.ai docs: temperature=0 is not valid for OpenAI-style calls; use a small epsilon.
        if t <= 0:
            return 0.01
        return min(float(t), 0.99)

    def generate(self, prompt: str) -> LLMResponse:
        completion = self._client.chat.completions.create(
            model=os.getenv("ZAI_MODEL") or self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._effective_temperature(self.config.temperature),
            max_tokens=_zai_max_output_tokens(self.config),
            extra_body=_zai_extra_body(),
        )
        msg = completion.choices[0].message
        return LLMResponse(text=msg.content or "", thoughts=None)


class RotatingZaiLLM(BaseLLM):
    provider: ProviderName = "zai"

    def __init__(self, api_keys: List[str], config: Optional[ZaiConfig] = None):
        keys = [k.strip() for k in api_keys if k and k.strip()]
        if not keys:
            raise ValueError("Provide at least one Z.ai API key.")

        self.api_keys = keys
        self.config = config or ZaiConfig()
        self._idx = 0
        base = os.getenv("ZAI_BASE_URL") or self.config.base_url

        from openai import OpenAI  # lazy import

        self._OpenAI = OpenAI
        self._base_url = base
        self._client = self._OpenAI(api_key=self.api_keys[self._idx], base_url=base)

    def _rotate(self) -> None:
        self._idx = (self._idx + 1) % len(self.api_keys)
        self._client = self._OpenAI(api_key=self.api_keys[self._idx], base_url=self._base_url)

    def generate(self, prompt: str) -> LLMResponse:
        last_exc: Optional[Exception] = None
        for _ in range(len(self.api_keys)):
            try:
                completion = self._client.chat.completions.create(
                    model=os.getenv("ZAI_MODEL") or self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=ZaiLLM._effective_temperature(self.config.temperature),
                    max_tokens=_zai_max_output_tokens(self.config),
                    extra_body=_zai_extra_body(),
                )
                msg = completion.choices[0].message
                return LLMResponse(text=msg.content or "", thoughts=None)
            except Exception as e:
                if _should_rotate_key(e):
                    last_exc = e
                    self._rotate()
                    continue
                raise
        raise last_exc or RuntimeError("All Z.ai API keys failed.")


def build_llm(provider: ProviderName) -> BaseLLM:
    if provider == "groq":
        keys = _split_keys(os.getenv("GROQ_API_KEYS"))
        if keys:
            return RotatingGroqLLM(api_keys=keys)
        return GroqLLM()
    if provider == "gemini":
        keys = _split_keys(os.getenv("GEMINI_API_KEYS"))
        if keys:
            return RotatingGeminiLLM(api_keys=keys)
        return GeminiLLM()
    if provider == "zai":
        keys = _split_keys(os.getenv("ZAI_API_KEYS"))
        if keys:
            return RotatingZaiLLM(api_keys=keys)
        return ZaiLLM()
    raise ValueError(f"Unknown provider: {provider}")

