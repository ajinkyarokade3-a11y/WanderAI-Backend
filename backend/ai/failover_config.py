"""Configuration structures for the AI failover hierarchy.

Hierarchy (fixed order, sequential only)::

    Provider -> API Key -> Model

Provider priority: Gemini (1) -> OpenRouter (2) -> Grok (3).

This module contains **structures only** — no routing logic, no network
calls. The future router will consume :func:`build_failover_plan` and
iterate it strictly in order, stopping on the first success.

Model-ID provenance (verified Sep 2026, not invented):

* Gemini — https://ai.google.dev/gemini-api/docs/models and
  https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions
  (``gemini-2.5-flash`` stable primary; ``gemini-2.5-flash-lite``,
  ``gemini-3.1-flash-lite``, ``gemini-3.5-flash-lite``,
  ``gemini-3.5-flash`` GA with 12-month availability). ``gemini-2.0-*``
  are shut down (Jun 2026) and ``gemini-3.6/3.7/3.8-flash`` do not appear
  in the public docs, so they are NOT defaults here.
* OpenRouter — https://openrouter.ai/collections/free-models plus the
  Aug/Sep 2026 ``:free`` catalog snapshots. ``:free`` suffix is required;
  the free roster churns, so the list is env-overridable.
* Grok (xAI) — https://docs.x.ai/developers/models (Sep 2026). xAI has no
  ongoing $0 tier, so defaults are the cheapest documented IDs
  (``grok-4-1-fast-reasoning``, ``grok-code-fast-1``, ``grok-4.3``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# Fixed provider priority. Do not reorder without a deliberate migration.
PROVIDER_ORDER: List[str] = ["gemini", "openrouter", "grok"]

PROVIDER_PRIORITY: dict[str, int] = {
    "gemini": 1,
    "openrouter": 2,
    "grok": 3,
}

# Documented fallbacks used when the corresponding env CSV is blank.
# (Env values always win; these only guarantee a sane, verified default.)
DEFAULT_GEMINI_MODELS: List[str] = [
    "gemini-2.5-flash",  # primary
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
]

DEFAULT_OPENROUTER_MODELS: List[str] = [
    "openai/gpt-oss-20b:free",  # primary
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "cohere/north-mini-code:free",
]

DEFAULT_GROK_MODELS: List[str] = [
    # Primary is grok-4.3: verified Sep 2026 on the standard xAI pricing page
    # (https://docs.x.ai/developers/models, 1M context, Chat Completions
    # supported). grok-code-fast-1 is a documented alias (canonical
    # grok-build-0.1). grok-4-1-fast-reasoning stays as a fallback: xAI cites
    # "Grok 4.1 Fast" for the Enterprise API, so it may not serve standard keys
    # (a 404 there maps to MODEL_UNAVAILABLE and is skipped gracefully).
    "grok-4.3",  # primary
    "grok-code-fast-1",
    "grok-4-1-fast-reasoning",
]


def parse_model_list(raw: str | None, defaults: List[str]) -> List[str]:
    """Parse a comma-separated model CSV into an ordered, deduped list.

    Index 0 is the primary model; the rest are fallbacks in order.
    Blank input returns a copy of ``defaults``. Entries are stripped and
    empty items / duplicates (keeping first occurrence) are dropped.
    """
    if not (raw or "").strip():
        return list(defaults)
    seen: set[str] = set()
    ordered: List[str] = []
    for item in raw.split(","):
        name = item.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered or list(defaults)


@dataclass(frozen=True)
class ApiKeySlot:
    """One API-key slot within a provider (1-based ``slot``)."""

    slot: int  # 1, 2, or 3
    api_key: str = ""
    enabled: bool = True

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip()) and self.enabled


@dataclass(frozen=True)
class ProviderConfig:
    """Static configuration for one provider in the failover chain."""

    name: str  # "gemini" | "openrouter" | "grok"
    priority: int
    enabled: bool
    base_url: str
    keys: List[ApiKeySlot] = field(default_factory=list)
    models: List[str] = field(default_factory=list)

    @property
    def primary_model(self) -> str | None:
        return self.models[0] if self.models else None

    @property
    def fallback_models(self) -> List[str]:
        return list(self.models[1:])

    @property
    def configured_keys(self) -> List[ApiKeySlot]:
        return [k for k in self.keys if k.is_configured]


@dataclass(frozen=True)
class FailoverPlan:
    """Ordered provider configs plus shared resilience tuning for the router."""

    providers: List[ProviderConfig]  # already sorted by priority
    model_cooldown_s: float = 300.0
    key_cooldown_s: float = 600.0
    provider_cooldown_s: float = 900.0
    key_disable_s: float = 3600.0
    request_timeout_s: float = 30.0
    max_transient_retries: int = 1
    initial_backoff_ms: int = 1000
    max_backoff_ms: int = 10000

    @property
    def enabled_providers(self) -> List[ProviderConfig]:
        return [p for p in self.providers if p.enabled]


def _gemini_keys(settings) -> List[ApiKeySlot]:
    # Legacy GEMINI_API_KEY acts as an alias for slot 1 when _1 is empty,
    # preserving backward compatibility with existing deployments.
    key1 = (getattr(settings, "GEMINI_API_KEY_1", "") or "").strip() or (
        getattr(settings, "GEMINI_API_KEY", "") or ""
    ).strip()
    return [
        ApiKeySlot(slot=1, api_key=key1),
        ApiKeySlot(slot=2, api_key=(getattr(settings, "GEMINI_API_KEY_2", "") or "").strip()),
        ApiKeySlot(slot=3, api_key=(getattr(settings, "GEMINI_API_KEY_3", "") or "").strip()),
    ]


def _openrouter_keys(settings) -> List[ApiKeySlot]:
    return [
        ApiKeySlot(slot=i, api_key=(getattr(settings, f"OPENROUTER_API_KEY_{i}", "") or "").strip())
        for i in (1, 2, 3)
    ]


def _grok_keys(settings) -> List[ApiKeySlot]:
    return [
        ApiKeySlot(slot=i, api_key=(getattr(settings, f"GROK_API_KEY_{i}", "") or "").strip())
        for i in (1, 2, 3)
    ]


def build_failover_plan(settings=None) -> FailoverPlan:
    """Build the ordered Provider -> Key -> Model plan from settings/env.

    Accepts the project ``Settings`` instance (defaults to the global
    ``backend.database.config.settings``). Pure construction — no I/O,
    no network, no state.
    """
    if settings is None:
        from backend.database.config import settings as global_settings

        settings = global_settings

    providers = [
        ProviderConfig(
            name="gemini",
            priority=PROVIDER_PRIORITY["gemini"],
            enabled=bool(getattr(settings, "GEMINI_ENABLED", True)),
            base_url=str(getattr(settings, "GEMINI_BASE_URL", "") or ""),
            keys=_gemini_keys(settings),
            models=parse_model_list(
                getattr(settings, "GEMINI_MODELS", ""), DEFAULT_GEMINI_MODELS
            ),
        ),
        ProviderConfig(
            name="openrouter",
            priority=PROVIDER_PRIORITY["openrouter"],
            enabled=bool(getattr(settings, "OPENROUTER_ENABLED", True)),
            base_url=str(getattr(settings, "OPENROUTER_BASE_URL", "") or ""),
            keys=_openrouter_keys(settings),
            models=parse_model_list(
                getattr(settings, "OPENROUTER_MODELS", ""), DEFAULT_OPENROUTER_MODELS
            ),
        ),
        ProviderConfig(
            name="grok",
            priority=PROVIDER_PRIORITY["grok"],
            enabled=bool(getattr(settings, "GROK_ENABLED", True)),
            base_url=str(getattr(settings, "GROK_BASE_URL", "") or ""),
            keys=_grok_keys(settings),
            models=parse_model_list(
                getattr(settings, "GROK_MODELS", ""), DEFAULT_GROK_MODELS
            ),
        ),
    ]
    providers.sort(key=lambda p: p.priority)

    return FailoverPlan(
        providers=providers,
        model_cooldown_s=float(getattr(settings, "AI_MODEL_COOLDOWN_S", 300.0)),
        key_cooldown_s=float(getattr(settings, "AI_KEY_COOLDOWN_S", 600.0)),
        provider_cooldown_s=float(getattr(settings, "AI_PROVIDER_COOLDOWN_S", 900.0)),
        key_disable_s=float(getattr(settings, "AI_KEY_DISABLE_S", 3600.0)),
        request_timeout_s=float(getattr(settings, "AI_REQUEST_TIMEOUT_S", 30.0)),
        max_transient_retries=int(getattr(settings, "AI_MAX_TRANSIENT_RETRIES", 1)),
        initial_backoff_ms=int(getattr(settings, "AI_INITIAL_BACKOFF_MS", 1000)),
        max_backoff_ms=int(getattr(settings, "AI_MAX_BACKOFF_MS", 10000)),
    )
