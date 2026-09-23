"""Global AI failover router: ``Provider -> Key -> Model``, strictly sequential.

Fixed provider priority (from :mod:`backend.ai.failover_config`)::

    Request
      -> Gemini      (Key 1 -> models, Key 2 -> models, Key 3 -> models)
      -> OpenRouter  (Key 1 -> models, Key 2 -> models, Key 3 -> models)
      -> Grok        (Key 1 -> models, Key 2 -> models, Key 3 -> models)
      -> controlled service-unavailable error

Rules:

* One provider at a time — never concurrent. No ``asyncio.gather``, task
  racing, ``Promise.all``, threads, or any equivalent parallel fallback logic;
  this module is plain sequential control flow (a ``for`` loop with an
  immediate ``return``).
* Before attempting a provider, its cooldown gate is checked
  (``HealthManager.provider_available``); cooled/disabled providers are
  skipped without any upstream call. Key/model gates live inside
  :func:`cascade_provider`, which owns the per-provider traversal.
* The moment any model returns a valid response, it is returned at once —
  no further providers, keys, or models are called for that request.
* A provider cascade that raises a non-cross-provider error
  (``INVALID_REQUEST`` / ``CONTENT_POLICY_ERROR`` — see
  :func:`directive_for`) is re-raised immediately instead of cycling the
  remaining providers: caller-side and policy errors must not burn other
  providers.
* If every provider is skipped or exhausted, a single controlled
  service-unavailable :class:`AIProviderError` is raised: its message is the
  generic user-safe text (no provider names, no raw provider errors, no key
  material) so API layers can surface it directly, while the tried-provider
  list and the last failure stay in the structured logs and the ``__cause__``
  chain for operators.

The router returns :class:`AIResponse` (``text``/``provider``/``model``) and
never shapes endpoint payloads — Traveller-App and Operator-Web observe no
API-contract change regardless of which provider generated the response.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

from backend.ai.ai_types import (
    AIErrorCategory,
    AIProviderError,
    AIRequest,
    AIResponse,
    public_message_for,
)
from backend.ai.errors import directive_for
from backend.ai.health import HealthManager
from backend.ai.observability import (
    get_ai_logger,
    log_provider_attempt,
    log_provider_failed,
    log_provider_skipped,
    log_request_failed,
    log_request_started,
    new_request_id,
)

logger = get_ai_logger()

__all__ = ["generate_with_failover", "PROVIDER_PRIORITY_ORDER"]


#: Canonical attempt order. Mirrors ``failover_config.PROVIDER_ORDER``;
#: the plan's priority-sorted providers are the source of truth at runtime
#: and must agree with this list (asserted in tests).
PROVIDER_PRIORITY_ORDER: List[str] = ["gemini", "openrouter", "grok"]


def generate_with_failover(
    request: AIRequest,
    *,
    plan=None,
    health: Optional[HealthManager] = None,
    sleep: Callable[[float], None] = time.sleep,
    request_id: Optional[str] = None,
) -> AIResponse:
    """Attempt providers in priority order; return the first valid response.

    :param request: provider-agnostic request (same object for every attempt).
        Its prompt text is never logged — only identifiers and outcomes.
    :param plan: :class:`FailoverPlan` (defaults to ``build_failover_plan()``).
    :param health: :class:`HealthManager` (defaults to one built from the plan).
    :param sleep: backoff sleeper forwarded to the per-provider cascades.
    :param request_id: correlation ID shared with the cascades (generated
        when omitted).
    :raises AIProviderError: terminal per-provider failure with
        ``cross_provider=False`` (passed through untouched), or a controlled
        ``PROVIDER_UNAVAILABLE`` service-unavailable error with a generic
        user-safe message when everything was skipped or exhausted.
    """
    from backend.ai.cascade import cascade_provider
    from backend.ai.failover_config import build_failover_plan

    plan = plan or build_failover_plan()
    health = health or HealthManager.from_plan(plan)
    request_id = request_id or new_request_id()
    ordered = sorted(plan.providers, key=lambda p: p.priority)
    log_request_started(logger, request_id, [p.name for p in ordered if p.enabled])
    started_all = time.perf_counter()
    attempted: List[str] = []
    last_error: Optional[AIProviderError] = None

    for cfg in ordered:
        name = cfg.name
        if not cfg.enabled:
            log_provider_skipped(logger, request_id, name, reason="disabled")
            continue  # disabled provider: no upstream call, no state change
        if not health.provider_available(name):
            log_provider_skipped(logger, request_id, name, reason="provider_cooldown")
            continue  # cooled provider: skip silently for this request
        attempted.append(name)
        log_provider_attempt(logger, request_id, name)
        try:
            # Sequential: this call runs the provider's full Key -> Model
            # cascade to completion (success or raise) before the loop moves on.
            return cascade_provider(name, request, plan=plan, health=health,
                                    sleep=sleep, request_id=request_id)
        except AIProviderError as exc:
            last_error = exc
            directive = directive_for(exc.category)
            log_provider_failed(logger, request_id, name, error=exc.category.value,
                                action=directive.action.value)
            # Caller-side / policy failures must not cycle other providers.
            if not directive.cross_provider:
                raise
            continue  # next provider in priority order

    latency_ms = (time.perf_counter() - started_all) * 1000.0
    tried = ", ".join(attempted) if attempted else "none"
    log_request_failed(logger, request_id, tried=tried,
                       error=(last_error.category.value if last_error else "all_skipped"),
                       latency_ms=latency_ms)
    # User-safe terminal error: generic text only. Tried-provider detail and
    # the raw provider failure stay in the logs above and the cause chain —
    # never in client-facing responses. API layers may use `.public_message`.
    terminal = AIProviderError(
        public_message_for(AIErrorCategory.PROVIDER_UNAVAILABLE),
        category=AIErrorCategory.PROVIDER_UNAVAILABLE,
        provider="",
        model="",
        retryable=True,
    )
    if last_error is not None:
        raise terminal from last_error
    raise terminal
