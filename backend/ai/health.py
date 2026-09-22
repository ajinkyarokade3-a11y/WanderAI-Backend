"""Centralized health-state / cooldown manager for AI failover resources.

Tracks three resource scopes — **providers**, **API keys**, **models** — each
with at minimum::

    status, cooldown_until, consecutive_failures, last_error, last_success

This module performs **no I/O, no retries, no fallback loops**. It answers:

* "may the router use this provider / key / model right now?" (with lazy
  expiry: a resource whose cooldown has passed is automatically eligible
  again — transient failures never permanently disable anything), and
* "record this success / failure" (applying the single-step
  :class:`FailoverDirective` from :mod:`backend.ai.errors`).

Scope isolation (the required behavior):

* ``429`` on ``gemini key 1`` cools down **only** ``key:gemini:1`` — keys 2/3
  and every model stay eligible.
* ``model unavailable`` on ``(gemini, key 1, model A)`` cools down **only**
  ``model:gemini:<A>`` — other Gemini models stay eligible on the same key.
* A provider-wide outage cools down ``provider:gemini`` so the router skips
  Gemini entirely for subsequent requests. A *single* 5xx must NOT do this:
  the classifier reports ``SERVER_ERROR`` for one bad call and the router
  promotes to a provider cooldown only after consecutive provider-wide
  failures (visible via ``consecutive_failures`` on the provider scope) or an
  explicit ``PROVIDER_UNAVAILABLE`` classification.

Storage design (Redis-ready):

* The manager talks only to the tiny :class:`HealthStore` interface
  (``get`` / ``update`` / ``clear``). The default :class:`InMemoryHealthStore`
  keeps everything in-process, which fits the current single-process FastAPI
  deployment.
* A future shared store (Redis, …) implements the same three methods —
  ``update`` as an atomic read-modify-write (e.g. a Lua script or WATCH/MULTI
  block) — and the manager/router code stays untouched.

Concurrency:

* All methods are synchronous, non-blocking, and hold only a brief in-memory
  lock (inside the store). That makes them safe to call from both threaded
  workers and ``async`` FastAPI handlers (no ``await`` needed, no
  loop-bound ``asyncio.Lock``). Do NOT add blocking I/O inside these paths.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional

from backend.ai.ai_types import AIErrorCategory, AIProviderError
from backend.ai.errors import classify_exception, directive_for

__all__ = [
    "ResourceStatus",
    "ErrorRecord",
    "ResourceState",
    "HealthStore",
    "InMemoryHealthStore",
    "HealthManager",
    "key_scope",
    "model_scope",
    "provider_scope",
]


class ResourceStatus(str, Enum):
    """Lifecycle state of one tracked scope (derived from expiry timestamps)."""

    HEALTHY = "healthy"  # eligible for use
    COOLDOWN = "cooldown"  # temporarily skipped until ``cooldown_until``
    DISABLED = "disabled"  # key rejected by provider: skipped until ``disabled_until``


@dataclass(frozen=True)
class ErrorRecord:
    """Compact, leak-free snapshot of the last failure on a scope."""

    message: str  # truncated provider message (never key material)
    category: str  # AIErrorCategory value
    at: float  # wall-clock seconds when recorded


@dataclass
class ResourceState:
    """Tracked health for one provider / key / model scope.

    ``status`` is maintained on every write and lazily refreshed on read, so
    expiry automatically restores eligibility without any sweeper thread.
    ``last_success`` is ``0.0`` when no success was ever recorded.
    """

    status: ResourceStatus = ResourceStatus.HEALTHY
    cooldown_until: float = 0.0
    disabled_until: float = 0.0
    consecutive_failures: int = 0
    last_error: Optional[ErrorRecord] = None
    last_success: float = 0.0

    def is_available(self, now: float) -> bool:
        """True when neither a cooldown nor a disable window covers ``now``."""
        return now >= self.cooldown_until and now >= self.disabled_until

    def refresh(self, now: float) -> "ResourceState":
        """Re-derive ``status`` from expiry timestamps (idempotent)."""
        if self.is_available(now):
            if self.status is not ResourceStatus.HEALTHY:
                self.status = ResourceStatus.HEALTHY
        elif now < self.disabled_until:
            self.status = ResourceStatus.DISABLED
        else:
            self.status = ResourceStatus.COOLDOWN
        return self


# ---------------------------------------------------------------------------
# Scope keys (single canonical spelling per scope; models are case-sensitive
# provider model IDs, provider names are lowercased).
# ---------------------------------------------------------------------------

def provider_scope(provider: str) -> str:
    return f"provider:{(provider or '').strip().lower()}"


def key_scope(provider: str, slot: int) -> str:
    return f"key:{(provider or '').strip().lower()}:{int(slot)}"


def model_scope(provider: str, model: str) -> str:
    return f"model:{(provider or '').strip().lower()}:{(model or '').strip()}"


# ---------------------------------------------------------------------------
# Storage abstraction (in-memory today, Redis-compatible tomorrow).
# ---------------------------------------------------------------------------

class HealthStore(ABC):
    """Minimal atomic state backend. All methods are non-blocking."""

    @abstractmethod
    def get(self, scope: str) -> Optional[ResourceState]:
        """Return a *copy* of the stored state, or None if never recorded."""
        raise NotImplementedError

    @abstractmethod
    def update(
        self, scope: str, fn: Callable[[ResourceState], ResourceState]
    ) -> ResourceState:
        """Atomically apply ``fn`` to the state for ``scope`` (creating a
        fresh :class:`ResourceState` when absent) and return a copy of the
        result. A Redis implementation must make this a single atomic step
        (Lua script / WATCH-MULTI); the manager performs no other
        read-modify-write of its own."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Drop all tracked state (tests / operator reset)."""
        raise NotImplementedError


class InMemoryHealthStore(HealthStore):
    """Process-local store guarded by one re-entrant lock."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._states: Dict[str, ResourceState] = {}

    def get(self, scope: str) -> Optional[ResourceState]:
        with self._lock:
            state = self._states.get(scope)
            if state is None:
                return None
            return ResourceState(
                status=state.status,
                cooldown_until=state.cooldown_until,
                disabled_until=state.disabled_until,
                consecutive_failures=state.consecutive_failures,
                last_error=state.last_error,
                last_success=state.last_success,
            )

    def update(
        self, scope: str, fn: Callable[[ResourceState], ResourceState]
    ) -> ResourceState:
        with self._lock:
            current = self._states.get(scope)
            if current is None:
                current = ResourceState()
            result = fn(current)
            self._states[scope] = result
            return self.get(scope)  # type: ignore[return-value]

    def clear(self) -> None:
        with self._lock:
            self._states.clear()


# ---------------------------------------------------------------------------
# Manager (single-step records + eligibility checks; no loops).
# ---------------------------------------------------------------------------

def _short_message(exc: BaseException, limit: int = 200) -> str:
    text = exc.args[0] if exc.args and isinstance(exc.args[0], str) else str(exc)
    return (text or type(exc).__name__)[:limit]


class HealthManager:
    """Eligibility checks + success/failure accounting over a HealthStore.

    Durations are per-instance and configurable (prefer
    :meth:`from_plan`, which pulls ``AI_*_COOLDOWN_S`` from the
    :class:`FailoverPlan`). ``clock`` defaults to :func:`time.time` and is
    injectable for deterministic tests.
    """

    def __init__(
        self,
        store: Optional[HealthStore] = None,
        *,
        key_cooldown_s: float = 600.0,
        model_cooldown_s: float = 300.0,
        provider_cooldown_s: float = 900.0,
        key_disable_s: float = 3600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store = store or InMemoryHealthStore()
        self._key_cooldown_s = max(0.0, float(key_cooldown_s))
        self._model_cooldown_s = max(0.0, float(model_cooldown_s))
        self._provider_cooldown_s = max(0.0, float(provider_cooldown_s))
        self._key_disable_s = max(0.0, float(key_disable_s))
        self._clock = clock

    @classmethod
    def from_plan(cls, plan, store: Optional[HealthStore] = None, **overrides) -> "HealthManager":
        """Build from a :class:`FailoverPlan` (durations) plus overrides.

        ``key_disable_s`` defaults to ``AI_KEY_DISABLE_S`` on the plan when
        present, else 3600s. Explicit kwargs win over plan values.
        """
        kwargs = {
            "key_cooldown_s": float(getattr(plan, "key_cooldown_s", 600.0)),
            "model_cooldown_s": float(getattr(plan, "model_cooldown_s", 300.0)),
            "provider_cooldown_s": float(getattr(plan, "provider_cooldown_s", 900.0)),
            "key_disable_s": float(getattr(plan, "key_disable_s", 3600.0)),
        }
        kwargs.update(overrides)
        return cls(store, **kwargs)

    # -- eligibility (lazy expiry: no sweeper, no permanent bans) ---------

    def _available(self, scope: str) -> bool:
        now = self._clock()

        def check(state: ResourceState) -> ResourceState:
            return state.refresh(now)

        return self._store.update(scope, check).is_available(now)

    def provider_available(self, provider: str) -> bool:
        return self._available(provider_scope(provider))

    def key_available(self, provider: str, slot: int) -> bool:
        return self._available(key_scope(provider, slot))

    def model_available(self, provider: str, model: str) -> bool:
        return self._available(model_scope(provider, model))

    # -- state inspection (monitoring / router escalation signals) --------

    def _snapshot(self, scope: str) -> ResourceState:
        now = self._clock()
        state = self._store.get(scope) or ResourceState()
        return state.refresh(now)

    def provider_state(self, provider: str) -> ResourceState:
        return self._snapshot(provider_scope(provider))

    def key_state(self, provider: str, slot: int) -> ResourceState:
        return self._snapshot(key_scope(provider, slot))

    def model_state(self, provider: str, model: str) -> ResourceState:
        return self._snapshot(model_scope(provider, model))

    # -- explicit marks (router-driven; durations configurable) -----------

    def mark_key_cooldown(
        self, provider: str, slot: int, duration_s: Optional[float] = None
    ) -> ResourceState:
        return self._cool(key_scope(provider, slot),
                          self._key_cooldown_s if duration_s is None else duration_s)

    def mark_model_cooldown(
        self, provider: str, model: str, duration_s: Optional[float] = None
    ) -> ResourceState:
        return self._cool(model_scope(provider, model),
                          self._model_cooldown_s if duration_s is None else duration_s)

    def mark_provider_cooldown(
        self, provider: str, duration_s: Optional[float] = None
    ) -> ResourceState:
        return self._cool(provider_scope(provider),
                          self._provider_cooldown_s if duration_s is None else duration_s)

    def disable_key(
        self, provider: str, slot: int, duration_s: Optional[float] = None
    ) -> ResourceState:
        """Long-window skip for provider-rejected keys (NOT permanent)."""
        window = self._key_disable_s if duration_s is None else max(0.0, float(duration_s))
        now = self._clock()

        def apply(state: ResourceState) -> ResourceState:
            state.disabled_until = max(state.disabled_until, now + window)
            state.refresh(now)
            return state

        return self._store.update(key_scope(provider, slot), apply)

    def _cool(self, scope: str, duration_s: float) -> ResourceState:
        window = max(0.0, float(duration_s))
        now = self._clock()

        def apply(state: ResourceState) -> ResourceState:
            state.cooldown_until = max(state.cooldown_until, now + window)
            state.refresh(now)
            return state

        return self._store.update(scope, apply)

    # -- success / failure accounting -------------------------------------

    def record_success(self, provider: str, slot: int, model: str) -> None:
        """A working attempt proves all three scopes healthy: reset failures,
        stamp success, and lift any cooldown/disable windows."""
        now = self._clock()

        def healthy(state: ResourceState) -> ResourceState:
            state.consecutive_failures = 0
            state.last_success = now
            state.cooldown_until = 0.0
            state.disabled_until = 0.0
            state.status = ResourceStatus.HEALTHY
            return state

        self._store.update(provider_scope(provider), healthy)
        self._store.update(key_scope(provider, slot), healthy)
        self._store.update(model_scope(provider, model), healthy)

    def record_failure(
        self,
        provider: str,
        slot: int,
        model: str,
        error: BaseException,
    ):
        """Apply the :func:`directive_for` single step for one failure.

        Accepts an :class:`AIProviderError` (normal path) or any exception
        (normalized via :func:`classify_exception`). Every involved scope
        records the failure (``consecutive_failures += 1`` + ``last_error``)
        so the router can escalate on repeats; cooldown/disable windows are
        applied **only** to the scopes the directive names — never the whole
        provider for one transient blip. Returns the applied directive for
        the router's next-step decision.
        """
        if isinstance(error, AIProviderError):
            normalized = error
            category: AIErrorCategory = error.category
        else:
            normalized = classify_exception(error, provider=provider, model=model)
            category = normalized.category
        directive = directive_for(category)
        record = ErrorRecord(
            message=_short_message(normalized),
            category=category.value,
            at=self._clock(),
        )
        p_scope, k_scope, m_scope = (
            provider_scope(provider),
            key_scope(provider, slot),
            model_scope(provider, model),
        )

        def account(state: ResourceState) -> ResourceState:
            state.consecutive_failures += 1
            state.last_error = record
            return state

        # Accounting on all involved scopes (escalation signal), cooldowns
        # strictly per-directive (isolation).
        self._store.update(p_scope, account)
        self._store.update(k_scope, account)
        self._store.update(m_scope, account)
        if directive.cooldown_key:
            if directive.disable_key:
                self.disable_key(provider, slot)
            else:
                self.mark_key_cooldown(provider, slot)
        if directive.cooldown_model:
            self.mark_model_cooldown(provider, model)
        if directive.cooldown_provider:
            self.mark_provider_cooldown(provider)
        return directive

    # -- maintenance -------------------------------------------------------

    def reset(self) -> None:
        """Drop all tracked state (tests / operator reset)."""
        self._store.clear()

    def snapshot(self) -> Dict[str, ResourceState]:
        """Copy of every tracked scope (health endpoint / logging)."""
        store = self._store
        if isinstance(store, InMemoryHealthStore):
            with store._lock:
                return dict(store._states)
        return {}
