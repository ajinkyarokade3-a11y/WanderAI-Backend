"""Shared internal types for the AI provider abstraction.

The router only ever sees :class:`AIRequest`, :class:`AIResponse` and
:class:`AIProviderError`. Provider SDK / HTTP details must never leak past
the adapter boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AIErrorCategory(str, Enum):
    """Canonical normalized failure modes.

    The single-step failover decision for each member lives in
    :mod:`backend.ai.errors` (:func:`directive_for`). This enum carries no
    routing logic itself — it is just the shared vocabulary between adapters
    and the (future) router.

    Canonical members (new code must use these):

    * RATE_LIMITED — 429 / per-key rate limit. Next key + key cooldown.
    * QUOTA_EXCEEDED — billing/credit/quota exhausted. Next key + key cooldown.
    * INVALID_API_KEY — key rejected (401/invalid-key). Disable/heavy-cool the
      key, move to the next key. Never retry the same slot.
    * AUTHENTICATION_ERROR — 403 / forbidden / scope problems. Next key with a
      heavy key cooldown (may be account-wide; router escalates on repeats).
    * MODEL_UNAVAILABLE — 404 / unknown or delisted model. Cooldown the model,
      next model on the same key.
    * MODEL_OVERLOADED — capacity/overload on this model. Bounded retry, then
      next model + model cooldown.
    * PROVIDER_UNAVAILABLE — provider-wide outage signal. Next provider +
      provider cooldown.
    * TIMEOUT — deadline exceeded. Bounded retry, then next model.
    * NETWORK_ERROR — DNS / connection / transport failure. Bounded retry,
      then next model (router may escalate to next provider on repeats).
    * SERVER_ERROR — 5xx / malformed payload / empty reply. Bounded retry,
      then next model.
    * INVALID_REQUEST — 400/422 caller-side error. Must NOT fan out across
      providers: fail fast (at most a same-provider model retry), then surface.
    * CONTENT_POLICY_ERROR — safety/guardrail refusal. Fail fast, surface to
      the user. Never burn keys/models retrying it.
    * UNKNOWN_ERROR — unrecognized. Conservative transient treatment.

    The ``ALL_CAPS`` aliases below (``RATE_LIMIT``, ``AUTH_INVALID_KEY``,
    ``MODEL_NOT_FOUND``, ``BAD_REQUEST``, ``TRANSIENT``, ``UNKNOWN``) are
    deprecated backwards-compatibility aliases for code written against the
    earlier 7-member taxonomy. They are the *same enum members* (shared
    values), not separate categories.
    """

    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    INVALID_API_KEY = "invalid_api_key"
    AUTHENTICATION_ERROR = "authentication_error"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_OVERLOADED = "model_overloaded"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    SERVER_ERROR = "server_error"
    INVALID_REQUEST = "invalid_request"
    CONTENT_POLICY_ERROR = "content_policy_error"
    UNKNOWN_ERROR = "unknown_error"

    # --- Deprecated aliases (same values => same members) ---
    RATE_LIMIT = "rate_limited"
    AUTH_INVALID_KEY = "invalid_api_key"
    MODEL_NOT_FOUND = "model_unavailable"
    BAD_REQUEST = "invalid_request"
    TRANSIENT = "server_error"
    UNKNOWN = "unknown_error"

    @property
    def rotate_key(self) -> bool:
        """Whether the router should move to the next API key."""
        return self in (
            AIErrorCategory.RATE_LIMITED,
            AIErrorCategory.QUOTA_EXCEEDED,
            AIErrorCategory.INVALID_API_KEY,
            AIErrorCategory.AUTHENTICATION_ERROR,
        )

    @property
    def cooldown_model(self) -> bool:
        """Whether the router should cool down the failing model.

        Rate-limit/quota failures cool only the key: the model remains usable
        on the provider's other keys.
        """
        return self in (
            AIErrorCategory.MODEL_UNAVAILABLE,
            AIErrorCategory.MODEL_OVERLOADED,
            AIErrorCategory.SERVER_ERROR,
            AIErrorCategory.TIMEOUT,
            AIErrorCategory.NETWORK_ERROR,
        )


@dataclass(frozen=True)
class AIRequest:
    """Provider-agnostic generation request.

    ``prompt`` carries the full user content. Existing Gemini call sites
    inline their system instructions into the prompt string; ``system_instruction``
    is a separate optional field so OpenAI-compatible providers can send a
    real ``system`` message while Gemini concatenates it (same text either way).
    """

    prompt: str
    system_instruction: Optional[str] = None
    temperature: Optional[float] = None
    response_mime_type: Optional[str] = None  # "application/json" or None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    timeout_s: float = 30.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AIResponse:
    """Normalized success envelope returned by every adapter."""

    text: str
    provider: str  # "gemini" | "openrouter" | "grok"
    model: str


class AIProviderError(Exception):
    """Normalized adapter failure. Carries no provider SDK objects."""

    def __init__(
        self,
        message: str,
        *,
        category: AIErrorCategory = AIErrorCategory.UNKNOWN_ERROR,
        provider: str = "",
        model: str = "",
        status_code: Optional[int] = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.retryable = retryable

    def __str__(self) -> str:
        base = super().__str__()
        return f"[{self.provider}/{self.model} {self.category.value}] {base}"

    @property
    def public_message(self) -> str:
        """User-safe message: generic text with no provider/model/key names,
        no status codes, no error-category values, and no raw provider text.
        API layers must prefer this over ``str(exc)`` for client responses;
        the detailed message stays in logs and the ``__cause__`` chain."""
        return public_message_for(self.category)


def public_message_for(category: "AIErrorCategory") -> str:
    """Generic user-safe text for a category (no identifiers of any kind)."""
    if category is AIErrorCategory.INVALID_REQUEST:
        return "The AI request could not be processed. Please adjust your input and try again."
    if category is AIErrorCategory.CONTENT_POLICY_ERROR:
        return "The AI service could not respond to that request."
    return "The AI service is temporarily unavailable. Please try again shortly."
