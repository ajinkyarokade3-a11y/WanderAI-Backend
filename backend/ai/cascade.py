"""Sequential per-provider cascade: ``Provider -> Key -> Model``.

Hierarchy executed here (keys outer, models inner, strictly one upstream
call at a time — never parallel)::

    Provider
      Key 1 -> Model 1, Model 2, Model 3, ...
      Key 2 -> Model 1, Model 2, Model 3, ...
      Key 3 -> Model 1, Model 2, Model 3, ...

The PRIMARY model is always ``models[0]`` from the configured
:class:`ProviderConfig`; this module never reorders the list. Cooled-down
keys/models are skipped, unconfigured (empty) key slots are skipped, and the
first success returns immediately — later keys/models are never touched.

Per-attempt outcome handling (via :func:`directive_for` + :class:`HealthManager`,
which own classification and cooldown state — this module only sequences):

* success → ``record_success`` + return at once.
* ``RETRY_SAME_SLOT`` (timeout / transient / overload) → bounded same-slot
  retry (``plan.max_transient_retries``) with exponential backoff
  (``initial_backoff_ms`` → ``max_backoff_ms``). Retries happen *before*
  anything is recorded, so a blip that recovers never touches cooldown
  state; only the terminal failure is recorded (applying that directive's
  cooldowns) before moving to the next model.
* ``NEXT_MODEL`` → record (cools the model) + continue with the next model.
* ``NEXT_KEY`` → record (cools/disables the key) + move to the next key.
* ``NEXT_PROVIDER`` → record (cools the provider) + stop this cascade.
* ``FAIL_FAST`` (invalid request / policy refusal) → record accounting only
  (those directives carry no cooldowns) + raise at once; never cycles keys,
  models, or providers.

Exhaustion raises the last :class:`AIProviderError` so its category survives
for the outer router. When zero attempts are possible (provider disabled, no
configured keys, or everything on cooldown) it raises ``PROVIDER_UNAVAILABLE``
so the outer router moves to the next provider.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from backend.ai.ai_types import AIErrorCategory, AIProviderError, AIRequest, AIResponse
from backend.ai.errors import FailoverAction, directive_for
from backend.ai.health import HealthManager
from backend.ai.observability import (
    format_cooldowns,
    get_ai_logger,
    log_attempt,
    log_attempt_failed,
    log_attempt_success,
    new_request_id,
)

__all__ = [
    "cascade_provider",
    "generate_via_gemini",
    "generate_via_openrouter",
    "generate_via_grok",
]

logger = get_ai_logger()


def _backoff_s(retry_index: int, initial_ms: int, max_ms: int) -> float:
    initial = max(0, int(initial_ms))
    cap = max(0, int(max_ms))
    return min(initial * (2**retry_index), cap) / 1000.0


def _next_slot_number(slots, current_slot: int) -> str:
    """Next configured slot after ``current_slot`` for fallback reporting."""
    later = [k.slot for k in slots if k.slot > current_slot]
    return f"key={min(later)}" if later else "none"


def _next_model_name(models, health, provider: str, current: str) -> str:
    """Next available model after ``current`` for fallback reporting."""
    try:
        rest = models[models.index(current) + 1:]
    except ValueError:
        rest = []
    for name in rest:
        if health.model_available(provider, name):
            return f"model={name}"
    return "none"


def cascade_provider(
    provider: str,
    request: AIRequest,
    *,
    plan=None,
    health: Optional[HealthManager] = None,
    sleep: Callable[[float], None] = time.sleep,
    request_id: Optional[str] = None,
) -> AIResponse:
    """Run one provider's ``Key -> Model`` cascade sequentially.

    :param provider: ``"gemini"`` (also usable for later providers).
    :param request: provider-agnostic request (same object for every attempt).
    :param plan: :class:`FailoverPlan` (defaults to ``build_failover_plan()``).
    :param health: :class:`HealthManager` (defaults to one built from the plan).
    :param sleep: backoff sleeper (injectable no-op for tests).
    :param request_id: correlation ID (generated when omitted). The request
        prompt itself is never logged — only identifiers and outcomes.
    """
    from backend.ai import providers as provider_adapters
    from backend.ai.failover_config import build_failover_plan

    want = (provider or "").strip().lower()
    plan = plan or build_failover_plan()
    health = health or HealthManager.from_plan(plan)
    cfg = next((p for p in plan.providers if p.name == want), None)

    if cfg is None or not cfg.enabled:
        raise AIProviderError(
            f"provider {want!r} is not enabled",
            category=AIErrorCategory.PROVIDER_UNAVAILABLE,
            provider=want, model="", retryable=True,
        )
    slots = [k for k in cfg.keys if k.is_configured]
    if not slots or not cfg.models:
        raise AIProviderError(
            f"provider {want!r} has no usable keys/models",
            category=AIErrorCategory.PROVIDER_UNAVAILABLE,
            provider=want, model="", retryable=True,
        )

    max_retries = max(0, int(getattr(plan, "max_transient_retries", 1)))
    request_id = request_id or new_request_id()
    last_error: Optional[AIProviderError] = None
    stop_provider = False
    attempt = 0

    for key_slot in slots:
        if stop_provider or not health.provider_available(want):
            break  # cooled mid-cascade: stop trying this provider
        if not health.key_available(want, key_slot.slot):
            logger.debug(
                "Skipping request_id=%s provider=%s key=%s reason=key_cooldown",
                request_id, want, key_slot.slot,
            )
            continue
        api_key = key_slot.api_key
        next_key = False
        for model in cfg.models:  # configured priority order; [0] is primary
            if stop_provider or not health.provider_available(want):
                stop_provider = True
                break
            if not health.model_available(want, model):
                logger.debug(
                    "Skipping request_id=%s provider=%s key=%s model=%s reason=model_cooldown",
                    request_id, want, key_slot.slot, model,
                )
                continue
            # --- one slot, bounded transient retries, one call at a time ---
            retries_used = 0
            while True:
                attempt += 1
                log_attempt(logger, request_id, want, key_slot.slot, model, attempt)
                started = time.perf_counter()
                try:
                    response = provider_adapters.generate(
                        want, api_key, model, request
                    )
                except AIProviderError as exc:
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    last_error = exc
                    # Peek at the directive first: decisive actions record
                    # immediately, while RETRY_SAME_SLOT retries *before*
                    # recording (a recovered blip leaves no cooldown trace).
                    directive = directive_for(exc.category)
                    action = directive.action
                    cooldowns = format_cooldowns(
                        directive.cooldown_key, directive.cooldown_model,
                        directive.cooldown_provider, directive.disable_key,
                    )
                    if action is FailoverAction.FAIL_FAST:
                        health.record_failure(want, key_slot.slot, model, exc)
                        log_attempt_failed(
                            logger, request_id, want, key_slot.slot, model, attempt,
                            error=exc.category.value, action=action.value,
                            cooldowns=cooldowns, fallback="none", latency_ms=latency_ms,
                        )
                        raise
                    if action is FailoverAction.NEXT_PROVIDER:
                        health.record_failure(want, key_slot.slot, model, exc)
                        log_attempt_failed(
                            logger, request_id, want, key_slot.slot, model, attempt,
                            error=exc.category.value, action=action.value,
                            cooldowns=cooldowns, fallback="next_provider",
                            latency_ms=latency_ms,
                        )
                        stop_provider = True
                        break  # out of while; flags exit the for loops
                    if action is FailoverAction.NEXT_KEY:
                        health.record_failure(want, key_slot.slot, model, exc)
                        log_attempt_failed(
                            logger, request_id, want, key_slot.slot, model, attempt,
                            error=exc.category.value, action=action.value,
                            cooldowns=cooldowns,
                            fallback=_next_slot_number(slots, key_slot.slot),
                            latency_ms=latency_ms,
                        )
                        next_key = True
                        break  # out of while; flag exits the model loop
                    if action is FailoverAction.NEXT_MODEL:
                        health.record_failure(want, key_slot.slot, model, exc)
                        log_attempt_failed(
                            logger, request_id, want, key_slot.slot, model, attempt,
                            error=exc.category.value, action=action.value,
                            cooldowns=cooldowns,
                            fallback=_next_model_name(cfg.models, health, want, model),
                            latency_ms=latency_ms,
                        )
                        break  # out of while; for advances (model cooled)
                    # RETRY_SAME_SLOT: same call again while budget remains.
                    if retries_used >= max_retries:
                        # Terminal failure: record (applies the directive's
                        # cooldowns) and move to the next model.
                        health.record_failure(want, key_slot.slot, model, exc)
                        log_attempt_failed(
                            logger, request_id, want, key_slot.slot, model, attempt,
                            error=exc.category.value, action=action.value,
                            cooldowns=cooldowns,
                            fallback=_next_model_name(cfg.models, health, want, model),
                            latency_ms=latency_ms,
                        )
                        break
                    log_attempt_failed(
                        logger, request_id, want, key_slot.slot, model, attempt,
                        error=exc.category.value, action=action.value,
                        cooldowns="none",
                        fallback=f"same_slot retries_left={max_retries - retries_used}",
                        latency_ms=latency_ms,
                    )
                    sleep(_backoff_s(retries_used, plan.initial_backoff_ms,
                                     plan.max_backoff_ms))
                    retries_used += 1
                    # Re-check gates before the retry (a concurrent request
                    # may have cooled this key/model/provider meanwhile).
                    if (not health.provider_available(want)
                            or not health.key_available(want, key_slot.slot)
                            or not health.model_available(want, model)):
                        health.record_failure(want, key_slot.slot, model, exc)
                        break
                    continue
                latency_ms = (time.perf_counter() - started) * 1000.0
                health.record_success(want, key_slot.slot, model)
                log_attempt_success(
                    logger, request_id, want, key_slot.slot, model,
                    attempt=attempt, latency_ms=latency_ms,
                )
                return response
            if stop_provider or next_key:
                break  # out of the model loop (provider stop / next key)
        # Falling out of the model loop (exhausted or flagged) proceeds to
        # the next key; stop_provider exits the key loop via its gate.
    if last_error is not None:
        raise last_error
    raise AIProviderError(
        f"provider {want!r}: all keys/models on cooldown or unavailable",
        category=AIErrorCategory.PROVIDER_UNAVAILABLE,
        provider=want, model="", retryable=True,
    )


def generate_via_gemini(
    request: AIRequest,
    *,
    plan=None,
    health: Optional[HealthManager] = None,
    sleep: Callable[[float], None] = time.sleep,
    request_id: Optional[str] = None,
) -> AIResponse:
    """Gemini sequential cascade: ``Key 1 -> models… -> Key 2 -> …``.

    Primary model (``GEMINI_MODELS[0]``) is always attempted first per key.
    Returns on the first success; keys/models are never called in parallel.
    """
    return cascade_provider("gemini", request, plan=plan, health=health, sleep=sleep,
                            request_id=request_id)


def generate_via_openrouter(
    request: AIRequest,
    *,
    plan=None,
    health: Optional[HealthManager] = None,
    sleep: Callable[[float], None] = time.sleep,
    request_id: Optional[str] = None,
) -> AIResponse:
    """OpenRouter sequential cascade: ``Key 1 -> free models… -> Key 2 -> …``.

    Same traversal as Gemini — primary free model
    (``OPENROUTER_MODELS[0]``) first per key, first success stops the whole
    cascade, one upstream call at a time. Free-model differences (``:free``
    slugs, extra headers) live in :class:`OpenRouterAdapter`, never here.
    """
    return cascade_provider("openrouter", request, plan=plan, health=health, sleep=sleep,
                            request_id=request_id)


def generate_via_grok(
    request: AIRequest,
    *,
    plan=None,
    health: Optional[HealthManager] = None,
    sleep: Callable[[float], None] = time.sleep,
    request_id: Optional[str] = None,
) -> AIResponse:
    """Grok sequential cascade: ``Key 1 -> models… -> Key 2 -> …``.

    Same traversal as Gemini — primary model (``GROK_MODELS[0]``) first per
    key, first success stops the whole cascade, one call at a time. xAI
    differences (base URL, model IDs) live in :class:`GrokAdapter`, never here.
    """
    return cascade_provider("grok", request, plan=plan, health=health, sleep=sleep,
                            request_id=request_id)
