"""Secret-free structured logging for AI failover decisions.

Every line is ``key=value`` pairs so it is readable under ANY logging
config (no custom formatter needed). One ``request_id`` (see
:func:`new_request_id`) correlates all lines of a single user request across
providers, keys, and models.

What is recorded per attempt: request ID, provider, model, key **slot
number**, attempt number, result, error category, failover action, cooldowns
applied, fallback destination, latency.

What is NEVER recorded here (by construction, not just convention):

* full API keys — helpers accept only the integer ``key_slot``; the key
  string is never a parameter, so it cannot leak;
* ``Authorization`` headers or other secrets;
* user passwords;
* prompt / message content — helpers accept no ``AIRequest`` and no text;
  only counts/identifiers are logged.

Model names and provider names ARE logged: they are operator-chosen config
identifiers (from ``*_MODELS`` env), not secrets, and are required to explain
why a request moved. If a deployment treats model names as sensitive, route
this logger to a restricted sink — the payload shape stays the same.

Operational note: ``database/migrations/env.py`` calls
``logging.config.fileConfig("alembic.ini")``, which disables all
pre-existing loggers by default (that file only defines
root/sqlalchemy/alembic). That affects only processes that run migrations
in-process (deploy-time migrate, pytest sessions via ``tests/conftest.py``).
The serving process (``backend/main.py`` uses ``basicConfig`` and never calls
``fileConfig``) is unaffected.
"""

from __future__ import annotations

import logging
import uuid
from typing import List

AI_LOGGER_NAME = "tourflow.ai"


def get_ai_logger() -> logging.Logger:
    """Shared logger for all failover observability (cascade + router)."""
    return logging.getLogger(AI_LOGGER_NAME)


def new_request_id() -> str:
    """Short random correlation ID for one user request (12 hex chars)."""
    return uuid.uuid4().hex[:12]


def _kv(**pairs) -> str:
    return " ".join(f"{k}={v}" for k, v in pairs.items())


def format_cooldowns(
    cooldown_key: bool = False,
    cooldown_model: bool = False,
    cooldown_provider: bool = False,
    disable_key: bool = False,
) -> str:
    """Human-readable cooldown set, e.g. ``key+model``, ``key(disabled)``."""
    parts = []
    if cooldown_key:
        parts.append("key(disabled)" if disable_key else "key")
    if cooldown_model:
        parts.append("model")
    if cooldown_provider:
        parts.append("provider")
    return "+".join(parts) if parts else "none"


def log_request_started(logger: logging.Logger, request_id: str, providers: List[str]) -> None:
    logger.info("AI request started %s", _kv(request_id=request_id, providers=",".join(providers)))


def log_provider_attempt(logger: logging.Logger, request_id: str, provider: str) -> None:
    logger.info("Trying provider %s", _kv(request_id=request_id, provider=provider))


def log_provider_skipped(
    logger: logging.Logger, request_id: str, provider: str, reason: str
) -> None:
    logger.debug("Skipping provider %s", _kv(request_id=request_id, provider=provider, reason=reason))


def log_attempt(
    logger: logging.Logger,
    request_id: str,
    provider: str,
    key_slot: int,
    model: str,
    attempt: int,
) -> None:
    logger.info(
        "Trying %s",
        _kv(request_id=request_id, provider=provider, key=key_slot, model=model, attempt=attempt),
    )


def log_attempt_failed(
    logger: logging.Logger,
    request_id: str,
    provider: str,
    key_slot: int,
    model: str,
    attempt: int,
    *,
    error: str,
    action: str,
    cooldowns: str = "none",
    fallback: str = "none",
    latency_ms: float = 0.0,
) -> None:
    logger.warning(
        "Request failed %s",
        _kv(
            request_id=request_id,
            provider=provider,
            key=key_slot,
            model=model,
            attempt=attempt,
            error=error,
            action=action,
            cooldowns=cooldowns,
            fallback=fallback,
            latency_ms=f"{latency_ms:.1f}",
        ),
    )


def log_attempt_success(
    logger: logging.Logger,
    request_id: str,
    provider: str,
    key_slot: int,
    model: str,
    *,
    attempt: int,
    latency_ms: float = 0.0,
) -> None:
    logger.info(
        "Success %s",
        _kv(
            request_id=request_id,
            provider=provider,
            key=key_slot,
            model=model,
            attempt=attempt,
            latency_ms=f"{latency_ms:.1f}",
        ),
    )


def log_provider_failed(
    logger: logging.Logger,
    request_id: str,
    provider: str,
    *,
    error: str,
    action: str,
) -> None:
    logger.warning(
        "Provider failed %s",
        _kv(request_id=request_id, provider=provider, error=error, action=action),
    )


def log_request_failed(
    logger: logging.Logger,
    request_id: str,
    *,
    tried: str,
    error: str,
    latency_ms: float = 0.0,
) -> None:
    logger.error(
        "AI request failed %s",
        _kv(request_id=request_id, tried=tried, error=error, latency_ms=f"{latency_ms:.1f}"),
    )


__all__ = [
    "AI_LOGGER_NAME",
    "get_ai_logger",
    "new_request_id",
    "format_cooldowns",
    "log_request_started",
    "log_provider_attempt",
    "log_provider_skipped",
    "log_attempt",
    "log_attempt_failed",
    "log_attempt_success",
    "log_provider_failed",
    "log_request_failed",
]
