"""Provider adapters: one common ``generate`` shape over Gemini/OpenRouter/Grok.

Conceptual usage (what the future router does, strictly sequentially)::

    from backend.ai.providers import generate
    from backend.ai.ai_types import AIRequest

    response = generate(
        provider="gemini",
        api_key=key,
        model=model,
        request=AIRequest(prompt=..., temperature=0.2,
                          response_mime_type="application/json"),
    )

Rules for this module:

* Single attempt only — no loops, no retries, no parallelism. The router
  owns ``Provider -> Key -> Model`` sequencing, cooldowns and backoff.
* Every adapter returns :class:`AIResponse` or raises :class:`AIProviderError`.
  Provider SDK exceptions / HTTP payloads never escape this module.
* Gemini behavior is preserved: same ``google.genai`` SDK call
  (``client.models.generate_content``) with the same ``config`` keys the
  existing code uses (``temperature`` / ``response_mime_type`` / ``top_p`` /
  ``top_k`` / ``max_output_tokens``).
* OpenRouter and Grok are OpenAI-compatible ``POST {base_url}/chat/completions``
  calls via ``httpx`` (already a project dependency).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from backend.ai.ai_types import (
    AIErrorCategory,
    AIProviderError,
    AIRequest,
    AIResponse,
)
from backend.ai.errors import (
    classify_exception as _central_normalize,
    classify_http as _central_classify,
    extract_status as _central_extract,
)

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("gemini", "openrouter", "grok")


# ---------------------------------------------------------------------------
# Classification delegates (centralized in backend.ai.errors).
#
# These thin wrappers stay here so existing imports keep working; the mapping
# itself lives in errors.classify_http / errors.classify_exception, where the
# full provider-to-category table is documented. No routing loops here —
# single-call normalization only.
# ---------------------------------------------------------------------------

def _classify_status(status: Optional[int], message: str) -> AIErrorCategory:
    return _central_classify(status, message)


def _extract_status(exc: Exception) -> Optional[int]:
    return _central_extract(exc)


def _normalize_exception(exc: Exception, *, provider: str, model: str) -> AIProviderError:
    return _central_normalize(exc, provider=provider, model=model)


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------

class BaseProviderAdapter(ABC):
    """Single-attempt generator for one provider family."""

    provider_name: str = ""

    @abstractmethod
    def generate(
        self, api_key: str, model: str, request: AIRequest, **kwargs
    ) -> AIResponse:
        """One attempt. Returns AIResponse or raises AIProviderError."""
        raise NotImplementedError

    def _require(self, api_key: str, model: str, request: AIRequest) -> None:
        if not (api_key or "").strip():
            raise AIProviderError(
                "missing api_key", category=AIErrorCategory.INVALID_API_KEY,
                provider=self.provider_name, model=model or "",
                retryable=False,
            )
        if not (model or "").strip():
            raise AIProviderError(
                "missing model", category=AIErrorCategory.INVALID_REQUEST,
                provider=self.provider_name, model="",
                retryable=False,
            )
        if not (request.prompt or "").strip():
            raise AIProviderError(
                "empty prompt", category=AIErrorCategory.INVALID_REQUEST,
                provider=self.provider_name, model=model,
                retryable=False,
            )


# ---------------------------------------------------------------------------
# Gemini (google-genai SDK — preserves existing call semantics)
# ---------------------------------------------------------------------------

class GeminiAdapter(BaseProviderAdapter):
    provider_name = "gemini"

    def generate(
        self, api_key: str, model: str, request: AIRequest, **kwargs
    ) -> AIResponse:
        self._require(api_key, model, request)
        try:
            from google import genai
        except ImportError as exc:
            raise AIProviderError(
                "google-genai SDK is not installed",
                category=AIErrorCategory.UNKNOWN_ERROR, provider="gemini",
                model=model, retryable=False,
            ) from exc
        try:
            # Per-call client so the router can use any of the N configured
            # keys (the legacy singleton is pinned to a single key).
            client = genai.Client(api_key=api_key)
            contents = (
                f"{request.system_instruction}\n{request.prompt}"
                if request.system_instruction
                else request.prompt
            )
            # Same config keys the existing code passes; omit Nones so the
            # request matches legacy behavior (plain-text call => no config).
            config: dict = {}
            if request.temperature is not None:
                config["temperature"] = request.temperature
            if request.response_mime_type is not None:
                config["response_mime_type"] = request.response_mime_type
            if request.max_output_tokens is not None:
                config["max_output_tokens"] = request.max_output_tokens
            if request.top_p is not None:
                config["top_p"] = request.top_p
            if request.top_k is not None:
                config["top_k"] = request.top_k
            kwargs: dict = {"model": model, "contents": contents}
            if config:
                kwargs["config"] = config
            response = client.models.generate_content(**kwargs)
            text = getattr(response, "text", "") or ""
            if not text.strip():
                raise AIProviderError(
                    "empty response", category=AIErrorCategory.SERVER_ERROR,
                    provider="gemini", model=model,
                )
            return AIResponse(text=text, provider="gemini", model=model)
        except AIProviderError:
            raise
        except Exception as exc:
            raise _normalize_exception(exc, provider="gemini", model=model) from exc


# ---------------------------------------------------------------------------
# OpenAI-compatible base (OpenRouter + Grok share the wire shape)
# ---------------------------------------------------------------------------

class OpenAICompatibleAdapter(BaseProviderAdapter):
    """POST {base_url}/chat/completions via httpx. Subclasses fix identity."""

    provider_name = "openai_compatible"
    default_base_url = ""
    _extra_headers: dict[str, str] = {}

    def _base_url(self, override: Optional[str] = None) -> str:
        if (override or "").strip():
            return override.strip().rstrip("/")
        from backend.database.config import settings

        key = f"{self.provider_name.upper()}_BASE_URL"
        configured = (getattr(settings, key, "") or "").strip() or self.default_base_url
        return configured.rstrip("/")

    def generate(
        self,
        api_key: str,
        model: str,
        request: AIRequest,
        *,
        base_url: Optional[str] = None,
    ) -> AIResponse:
        self._require(api_key, model, request)
        try:
            import httpx
        except ImportError as exc:
            raise AIProviderError(
                "httpx is not installed",
                category=AIErrorCategory.UNKNOWN_ERROR, provider=self.provider_name,
                model=model, retryable=False,
            ) from exc
        url = self._base_url(base_url) + "/chat/completions"
        messages = []
        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})
        messages.append({"role": "user", "content": request.prompt})
        body: dict = {"model": model, "messages": messages}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            body["max_tokens"] = request.max_output_tokens
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.response_mime_type == "application/json":
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        headers.update(self._extra_headers or {})
        timeout = max(1.0, float(request.timeout_s or 30.0))
        try:
            # One synchronous attempt; no retries here (router's job).
            resp = httpx.post(url, json=body, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                category = _classify_status(resp.status_code, resp.text or "")
                raise AIProviderError(
                    (resp.text or f"HTTP {resp.status_code}")[:300],
                    category=category,
                    provider=self.provider_name, model=model,
                    status_code=resp.status_code,
                    retryable=category
                    not in (
                        AIErrorCategory.INVALID_REQUEST,
                        AIErrorCategory.CONTENT_POLICY_ERROR,
                    ),
                )
            data = resp.json()
            try:
                text = data["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, TypeError) as exc:
                raise AIProviderError(
                    "malformed chat-completions payload",
                    category=AIErrorCategory.SERVER_ERROR,
                    provider=self.provider_name, model=model,
                ) from exc
            if not text.strip():
                raise AIProviderError(
                    "empty response", category=AIErrorCategory.SERVER_ERROR,
                    provider=self.provider_name, model=model,
                )
            return AIResponse(text=text, provider=self.provider_name, model=model)
        except AIProviderError:
            raise
        except Exception as exc:
            # httpx.TimeoutException / ConnectError land here -> TIMEOUT/TRANSIENT
            # via message classification; status stays None.
            raise _normalize_exception(exc, provider=self.provider_name, model=model) from exc


class OpenRouterAdapter(OpenAICompatibleAdapter):
    provider_name = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"

    @property
    def _extra_headers(self) -> dict[str, str]:  # type: ignore[override]
        # Recommended by OpenRouter docs; harmless if APP_URL is local.
        try:
            from backend.database.config import settings

            return {
                "HTTP-Referer": (getattr(settings, "APP_URL", "") or "").strip(),
                "X-Title": (getattr(settings, "PROJECT_NAME", "") or "").strip(),
            }
        except Exception:
            return {}


class GrokAdapter(OpenAICompatibleAdapter):
    """xAI via Chat Completions (``POST {base}/chat/completions``).

    NOTE (Sep 2026): xAI marks Chat Completions as legacy in favor of the
    Responses API (``POST /v1/responses`` with ``input``/``output`` envelope),
    but Chat Completions is still served — grok-4.6 docs list both APIs, and
    ``max_tokens``/``messages`` remain valid there. A Responses-API migration
    belongs here inside this adapter (request shape + response parsing) and
    must not leak into the cascade/router. We send only widely-supported
    params (xAI rejects e.g. ``presencePenalty``/``stop`` on reasoning models).
    """

    provider_name = "grok"
    default_base_url = "https://api.x.ai/v1"


# ---------------------------------------------------------------------------
# Registry + facade (the router's only entry point)
# ---------------------------------------------------------------------------

_ADAPTERS: dict[str, BaseProviderAdapter] = {
    "gemini": GeminiAdapter(),
    "openrouter": OpenRouterAdapter(),
    "grok": GrokAdapter(),
}


def get_adapter(provider: str) -> BaseProviderAdapter:
    """Return the adapter for a supported provider name (case-insensitive)."""
    key = (provider or "").strip().lower()
    adapter = _ADAPTERS.get(key)
    if adapter is None:
        raise AIProviderError(
            f"unsupported provider: {provider!r}",
            category=AIErrorCategory.INVALID_REQUEST, provider=provider or "",
            model="", retryable=False,
        )
    return adapter


def generate(
    provider: str,
    api_key: str,
    model: str,
    request: AIRequest,
    **kwargs,
) -> AIResponse:
    """Single normalized attempt::

        adapter.generate(provider="gemini", api_key=key, model=model,
                         request=request)

    Exactly one provider call, synchronously. Never loops, retries, fans out,
    or falls over to another provider/key/model — sequencing is the router's
    job. Raises :class:`AIProviderError` on any failure.
    """
    return get_adapter(provider).generate(api_key, model, request, **kwargs)
