"""Centralized AI error classification.

Purpose: given *any* provider failure (HTTP status, SDK exception, transport
error), answer two questions for the (future) router without leaking
provider-specific details past the adapter boundary:

1. **What happened?** — one :class:`AIErrorCategory`.
2. **What single step next?** — one :class:`FailoverDirective`
   (retry same slot / next model / next key / next provider / fail fast),
   plus which cooldowns to apply.

This module performs **no I/O, no retries, no loops**. It classifies one
failure at a time; sequencing (``Provider -> Key -> Model``) stays in the
router. Cooldown *durations* come from :class:`FailoverPlan`
(``AI_*_COOLDOWN_S``); this module only says *which* scopes to cool down.

Failover-action summary (see :func:`directive_for` for the authoritative
table)::

    category              -> single step        + cooldowns
    RATE_LIMITED          -> NEXT_KEY           + key (+short model)
    QUOTA_EXCEEDED        -> NEXT_KEY           + key (long)
    INVALID_API_KEY       -> NEXT_KEY           + disable/heavy-cool key
    AUTHENTICATION_ERROR  -> NEXT_KEY           + heavy-cool key
    MODEL_UNAVAILABLE     -> NEXT_MODEL         + model
    MODEL_OVERLOADED      -> RETRY then NEXT_MODEL + model
    PROVIDER_UNAVAILABLE  -> NEXT_PROVIDER      + provider
    TIMEOUT               -> RETRY then NEXT_MODEL
    NETWORK_ERROR         -> RETRY then NEXT_MODEL (escalate on repeats)
    SERVER_ERROR          -> RETRY then NEXT_MODEL + model
    INVALID_REQUEST       -> FAIL_FAST (never cross provider boundary)
    CONTENT_POLICY_ERROR  -> FAIL_FAST (surface to user)
    UNKNOWN_ERROR         -> RETRY then NEXT_MODEL (conservative transient)

Provider-to-category mapping (authoritative; status wins, message refines):

=================  ============================================================
Signal             Category
=================  ============================================================
HTTP 429           RATE_LIMITED, unless the body/wording says quota, billing,
                   credit, insufficient funds/balance, or spend limit — then
                   QUOTA_EXCEEDED.
HTTP 401           INVALID_API_KEY. Gemini ``API_KEY_INVALID`` /
                   ``UNAUTHENTICATED``; OpenRouter/Grok ``invalid api key`` /
                   ``incorrect api key`` / ``invalid xai api key``.
HTTP 403           AUTHENTICATION_ERROR, except when the message explicitly
                   says the key itself is invalid/unknown — then
                   INVALID_API_KEY.
HTTP 404           MODEL_UNAVAILABLE when the message names a model/endpoint
                   (OpenRouter ``No endpoints found`` for a delisted ``:free``
                   id; xAI/Gemini ``model not found`` / ``NOT_FOUND``).
                   A bare 404 with no model wording is still MODEL_UNAVAILABLE
                   (unknown targets behave like missing models, not outages).
HTTP 400 / 422     INVALID_REQUEST, except safety wording (``safety``,
                   ``blocked``, ``harm``, ``policy``, ``content filter``,
                   ``guardrail``) — then CONTENT_POLICY_ERROR.
HTTP 402           QUOTA_EXCEEDED (OpenRouter billing/credit exhaustion).
HTTP 408           TIMEOUT.
HTTP 429-shaped    Gemini ``RESOURCE_EXHAUSTED`` (no numeric code) -> RATE_LIMITED.
SDK wording        ``overload`` / ``capacity`` / ``try again later`` on a
                   model call -> MODEL_OVERLOADED; provider-wide wording
                   (``service unavailable`` with provider scope, maintenance,
                   ``provider`` outage) or HTTP 502/503 without model wording
                   -> SERVER_ERROR on a single call (the router escalates to
                   PROVIDER_UNAVAILABLE after consecutive provider-wide
                   failures — a single 503 must NOT be treated as a full
                   provider outage).
HTTP 5xx           SERVER_ERROR (single call). See above for escalation note.
httpx/timeout      ``TimeoutException`` / ``asyncio.TimeoutError`` /
                   ``deadline exceeded`` / ``timed out`` -> TIMEOUT.
Transport          ``ConnectError`` / DNS / ``Connection refused|reset`` /
                   network unreachable -> NETWORK_ERROR.
Empty/malformed    Empty reply text or unparseable envelope -> SERVER_ERROR
                   (bounded retry, then next model).
Anything else      UNKNOWN_ERROR (conservative: retryable, next model).
=================  ============================================================

Deliberately NOT provider failures (router must not burn the chain on these):

* ``unsupported provider`` / ``missing model`` / ``empty prompt`` — local
  programming/config errors -> INVALID_REQUEST, ``retryable=False``.
* ``missing api_key`` — local config gap -> INVALID_API_KEY,
  ``retryable=False`` for that slot (router skips the slot, no cooldown).
* SDK-not-installed (``google-genai`` / ``httpx`` missing) -> UNKNOWN_ERROR,
  ``retryable=False`` (retrying cannot help; surface/ops issue).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from backend.ai.ai_types import AIErrorCategory, AIProviderError

__all__ = [
    "AIErrorCategory",
    "AIProviderError",
    "FailoverAction",
    "FailoverDirective",
    "directive_for",
    "classify_http",
    "extract_status",
    "classify_exception",
    "new_error",
]


class FailoverAction(str, Enum):
    """The single next step the router should take for one classified failure."""

    RETRY_SAME_SLOT = "retry_same_slot"  # bounded retry, same provider/key/model
    NEXT_MODEL = "next_model"  # same provider + key, next model (+ model cooldown)
    NEXT_KEY = "next_key"  # same provider, next key (+ key cooldown)
    NEXT_PROVIDER = "next_provider"  # next provider (+ provider cooldown)
    FAIL_FAST = "fail_fast"  # surface to user; do not continue the chain


@dataclass(frozen=True)
class FailoverDirective:
    """One-step router guidance for a classified failure (no loop inside)."""

    action: FailoverAction
    cooldown_key: bool = False
    cooldown_model: bool = False
    cooldown_provider: bool = False
    disable_key: bool = False  # key is unusable: skip it for a long window
    retryable: bool = True  # False => never retry this exact slot
    cross_provider: bool = True  # False => must NOT continue to another provider
    reason: str = ""


# ---------------------------------------------------------------------------
# Category -> directive table (authoritative).
# ---------------------------------------------------------------------------

_DIRECTIVES: dict[AIErrorCategory, FailoverDirective] = {
    AIErrorCategory.RATE_LIMITED: FailoverDirective(
        action=FailoverAction.NEXT_KEY,
        cooldown_key=True,
        # NOTE: no model cooldown here — the model stays usable on the other
        # keys (model scopes are per provider+model, so cooling it would
        # wrongly penalize healthy keys).
        reason="Per-key rate limit: a different key may still have budget.",
    ),
    AIErrorCategory.QUOTA_EXCEEDED: FailoverDirective(
        action=FailoverAction.NEXT_KEY,
        cooldown_key=True,  # long key cooldown (durations live in FailoverPlan)
        reason="Quota/billing exhausted on this key: try the next key.",
    ),
    AIErrorCategory.MODEL_QUOTA_EXCEEDED: FailoverDirective(
        action=FailoverAction.NEXT_MODEL,
        cooldown_model=True,  # cool the quota-exhausted model only
        retryable=False,  # same model+key is quota-blocked: move to next model
        reason="Per-model quota exhausted (Gemini free tier): the SAME key "
               "still works for other models, so try the next model.",
    ),
    AIErrorCategory.INVALID_API_KEY: FailoverDirective(
        action=FailoverAction.NEXT_KEY,
        cooldown_key=True,
        disable_key=True,  # heavily cool / skip: the key itself is rejected
        retryable=False,  # never retry the same slot
        reason="Credential rejected: disable the key, move to the next key.",
    ),
    AIErrorCategory.AUTHENTICATION_ERROR: FailoverDirective(
        action=FailoverAction.NEXT_KEY,
        cooldown_key=True,  # heavy cooldown; may be account-wide
        retryable=False,
        reason="Forbidden/scope failure: next key; router escalates to next "
        "provider after repeated auth failures on this provider.",
    ),
    AIErrorCategory.MODEL_UNAVAILABLE: FailoverDirective(
        action=FailoverAction.NEXT_MODEL,
        cooldown_model=True,
        retryable=False,  # same model+key can never succeed: move on
        reason="Model unknown/delisted: cool it down, reuse the key on the "
        "next model.",
    ),
    AIErrorCategory.MODEL_OVERLOADED: FailoverDirective(
        action=FailoverAction.RETRY_SAME_SLOT,
        cooldown_model=True,
        reason="Model at capacity: bounded retry, then next model + cooldown.",
    ),
    AIErrorCategory.PROVIDER_UNAVAILABLE: FailoverDirective(
        action=FailoverAction.NEXT_PROVIDER,
        cooldown_provider=True,
        reason="Provider-wide outage: cool the provider, move on.",
    ),
    AIErrorCategory.TIMEOUT: FailoverDirective(
        action=FailoverAction.RETRY_SAME_SLOT,
        reason="Deadline exceeded: bounded retry, then next model.",
    ),
    AIErrorCategory.NETWORK_ERROR: FailoverDirective(
        action=FailoverAction.RETRY_SAME_SLOT,
        reason="Transport failure: bounded retry, then next model; router "
        "escalates to next provider on consecutive network failures.",
    ),
    AIErrorCategory.SERVER_ERROR: FailoverDirective(
        action=FailoverAction.RETRY_SAME_SLOT,
        cooldown_model=True,
        reason="Server-side/empty/malformed reply: bounded retry, then next "
        "model + cooldown. A single 5xx is NOT a provider outage.",
    ),
    AIErrorCategory.INVALID_REQUEST: FailoverDirective(
        action=FailoverAction.FAIL_FAST,
        retryable=False,
        cross_provider=False,  # NEVER blindly cycle providers on a bad request
        reason="Caller-side error: do not burn other providers/keys; surface.",
    ),
    AIErrorCategory.CONTENT_POLICY_ERROR: FailoverDirective(
        action=FailoverAction.FAIL_FAST,
        retryable=False,
        cross_provider=False,
        reason="Safety refusal: never retry keys/models; surface to the user.",
    ),
    AIErrorCategory.UNKNOWN_ERROR: FailoverDirective(
        action=FailoverAction.RETRY_SAME_SLOT,
        reason="Unrecognized: conservative transient handling (bounded retry, "
        "then next model).",
    ),
}


def directive_for(category: AIErrorCategory) -> FailoverDirective:
    """Return the single-step directive for a category (pure lookup)."""
    return _DIRECTIVES.get(category, _DIRECTIVES[AIErrorCategory.UNKNOWN_ERROR])


# ---------------------------------------------------------------------------
# Classification (status wins, message refines; SDK-agnostic).
# ---------------------------------------------------------------------------

# NOTE: bare "exhausted" is deliberately absent — Gemini's generic
# ``RESOURCE_EXHAUSTED`` rate-limit signal contains it, and must stay
# RATE_LIMITED unless a billing/quota word pins it to QUOTA_EXCEEDED.
_QUOTA_WORDS = (
    "quota",
    "billing",
    "credit",
    "insufficient",
    "balance",
    "spend limit",
    "spending limit",
)
# Gemini per-model quota errors name the specific model (e.g. "model: gemini-3.6-flash").
# These are model-scoped, not key-scoped: the SAME key still works for other models,
# so the cascade must try NEXT_MODEL (same key) rather than NEXT_KEY (which wastes
# keys 2/3 on an identical quota wall, then falls off the provider entirely).
_MODEL_QUOTA_PATTERN = "model:"
_KEY_WORDS = (
    "invalid api key",
    "api_key_invalid",
    "api key invalid",
    "incorrect api key",
    "invalid xai api key",
    "invalid key",
    "unknown key",
    "unknown api key",
    "unauthenticated",
)
_SAFETY_WORDS = (
    "safety",
    "blocked",
    "harm",
    "policy",
    "content filter",
    "guardrail",
    "finish_reason",
)
_OVERLOAD_WORDS = ("overload", "capacity", "try again later", "too busy")
_PROVIDER_OUTAGE_WORDS = ("maintenance", "provider outage", "provider unavailable")


def classify_http(status: Optional[int], message: str) -> AIErrorCategory:
    """Map one (HTTP status, message) pair to a category.

    Documented precedence: explicit auth/key wording first, then 429/402,
    then 404-model, then safety-wording (beats generic 400), then 400/422,
    then timeout/network wording, then model-overload vs provider-outage
    wording for 5xx-shaped failures.
    """
    msg = (message or "").lower()

    # 1. Key/auth wording beats everything (providers echo 400 with key text).
    if any(w in msg for w in _KEY_WORDS):
        return AIErrorCategory.INVALID_API_KEY
    if status in (401, 403) or "unauthorized" in msg or "permission_denied" in msg or "forbidden" in msg:
        if status == 403 or "forbidden" in msg or "permission_denied" in msg or "scope" in msg:
            return AIErrorCategory.AUTHENTICATION_ERROR
        return AIErrorCategory.INVALID_API_KEY

    # 2. Rate-limit / quota.
    if status == 429 or "resource_exhausted" in msg or "rate_limit" in msg or "rate limit" in msg or "too many requests" in msg or " 429" in msg:
        if status == 429 and any(w in msg for w in _QUOTA_WORDS):
            # Per-model quota (Gemini free tier): the message names the model,
            # e.g. "Quota exceeded ... model: gemini-3.6-flash". This is a
            # model-scoped wall — the SAME key still works for other models,
            # so the cascade must try NEXT_MODEL, not NEXT_KEY.
            if _MODEL_QUOTA_PATTERN in msg:
                return AIErrorCategory.MODEL_QUOTA_EXCEEDED
            return AIErrorCategory.QUOTA_EXCEEDED
        if "resource_exhausted" in msg and any(w in msg for w in _QUOTA_WORDS):
            if _MODEL_QUOTA_PATTERN in msg:
                return AIErrorCategory.MODEL_QUOTA_EXCEEDED
            return AIErrorCategory.QUOTA_EXCEEDED
        return AIErrorCategory.RATE_LIMITED
    if status == 402 or (any(w in msg for w in ("billing", "credit", "insufficient")) and "model" not in msg):
        return AIErrorCategory.QUOTA_EXCEEDED

    # 3. Model availability.
    if status == 404 or "not_found" in msg or "model not found" in msg or "no such model" in msg or "no endpoints found" in msg:
        if status == 404 or "model" in msg or "endpoint" in msg:
            return AIErrorCategory.MODEL_UNAVAILABLE

    # 4. Safety wording beats generic 400 (Gemini blocks surface as 400).
    if any(w in msg for w in _SAFETY_WORDS):
        return AIErrorCategory.CONTENT_POLICY_ERROR
    if status in (400, 422) or "invalid_argument" in msg or "invalid argument" in msg or "bad request" in msg:
        return AIErrorCategory.INVALID_REQUEST

    # 5. Timeout / transport.
    if status == 408 or "deadline exceeded" in msg or "timed out" in msg or "timeout" in msg:
        return AIErrorCategory.TIMEOUT
    if any(
        w in msg
        for w in (
            "connecterror",
            "connection refused",
            "connection reset",
            "dns",
            "name resolution",
            "network unreachable",
            "connection error",
        )
    ):
        return AIErrorCategory.NETWORK_ERROR

    # 6. 5xx-shaped: model overload vs provider outage vs generic server error.
    # A single 5xx is NEVER classified as a provider outage here; the router
    # promotes to PROVIDER_UNAVAILABLE only on consecutive provider-wide
    # failures (see module docstring).
    if status is not None and 500 <= status <= 599:
        if any(w in msg for w in _OVERLOAD_WORDS):
            return AIErrorCategory.MODEL_OVERLOADED
        if "unavailable" in msg and "model" in msg:
            return AIErrorCategory.MODEL_OVERLOADED
        return AIErrorCategory.SERVER_ERROR
    if any(w in msg for w in _OVERLOAD_WORDS):
        return AIErrorCategory.MODEL_OVERLOADED
    if any(w in msg for w in _PROVIDER_OUTAGE_WORDS):
        return AIErrorCategory.PROVIDER_UNAVAILABLE
    if any(w in msg for w in ("unavailable", "internal error", "temporarily", "502", "503", "500")):
        return AIErrorCategory.SERVER_ERROR

    return AIErrorCategory.UNKNOWN_ERROR


def extract_status(exc: Exception) -> Optional[int]:
    """Best-effort numeric status from SDK/HTTP exception shapes.

    Covers ``status_code`` (httpx/OpenAI SDKs), ``status`` and ``code``
    (google-genai ``ApiError.code``), and ``exc.response.status_code``.
    Returns ``None`` when no numeric status is present.
    """
    for attr in ("status_code", "status", "code"):
        try:
            value = getattr(exc, attr, None)
        except Exception:
            continue
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    try:
        response = getattr(exc, "response", None)
    except Exception:
        response = None
    if response is not None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int) and 100 <= status <= 599:
            return status
    return None


_NON_RETRYABLE = frozenset(
    {
        AIErrorCategory.INVALID_API_KEY,
        AIErrorCategory.AUTHENTICATION_ERROR,
        AIErrorCategory.INVALID_REQUEST,
        AIErrorCategory.CONTENT_POLICY_ERROR,
        AIErrorCategory.MODEL_UNAVAILABLE,
    }
)


def new_error(
    message: str,
    *,
    category: AIErrorCategory,
    provider: str = "",
    model: str = "",
    status_code: Optional[int] = None,
    retryable: Optional[bool] = None,
) -> AIProviderError:
    """Build a normalized error with category-consistent ``retryable`` default."""
    if retryable is None:
        retryable = category not in _NON_RETRYABLE
    return AIProviderError(
        (message or "")[:300] or category.value,
        category=category,
        provider=provider,
        model=model,
        status_code=status_code,
        retryable=retryable,
    )


def classify_exception(exc: Exception, *, provider: str, model: str) -> AIProviderError:
    """Normalize any provider exception (single call, no retries/loops).

    Pass-through for already-normalized errors. Transport-type exceptions
    without a status (``httpx.TimeoutException``, ``asyncio.TimeoutError``,
    ``ConnectError``) are mapped by type name before message classification so
    a terse timeout never degrades to UNKNOWN_ERROR.
    """
    if isinstance(exc, AIProviderError):
        return exc
    type_name = type(exc).__name__.lower()
    if "timeout" in type_name or "timedout" in type_name:
        return new_error(str(exc), category=AIErrorCategory.TIMEOUT,
                         provider=provider, model=model,
                         status_code=extract_status(exc))
    if "connect" in type_name or "dns" in type_name or "network" in type_name:
        return new_error(str(exc), category=AIErrorCategory.NETWORK_ERROR,
                         provider=provider, model=model,
                         status_code=extract_status(exc))
    status = extract_status(exc)
    category = classify_http(status, str(exc))
    return new_error(str(exc), category=category, provider=provider, model=model,
                     status_code=status)
