"""Failover system tests: Provider -> Key -> Model, strictly sequential.

Covers success paths, error-specific routing, cooldown lifecycle, critical
invariants (single in-flight call per request, stop-on-success, skips,
secret-freedom), per-request sequentiality under concurrent load, and
robustness (missing keys, disabled providers, malformed config).

Pure unit tests: no database access. The repo ``conftest.py`` session fixture
still runs (migrations + seed); run with a sqlite ``DATABASE_URL`` override,
e.g. ``DATABASE_URL=sqlite:///./test_ai_tmp.db pytest tests/test_ai_failover.py``.
"""

import logging
import re
import threading
import time

import pytest

from backend.ai import providers as adapters
from backend.ai.ai_types import AIErrorCategory, AIProviderError, AIRequest, AIResponse
from backend.ai.cascade import generate_via_gemini
from backend.ai.failover_config import (
    ApiKeySlot,
    FailoverPlan,
    ProviderConfig,
    build_failover_plan,
)
from backend.ai.health import HealthManager
from backend.ai.observability import AI_LOGGER_NAME
from backend.ai.router import generate_with_failover

CANARY_KEY = "SECRET_CANARY_KEY_9f8e7d6c5b"
CANARY_PROMPT = "CANARY_PROMPT_CONTENT_TOPSECRET_XQZ"

PRIORITY = {"gemini": 1, "openrouter": 2, "grok": 3}


class Clock:
    def __init__(self):
        self.now = 7_000_000.0

    def __call__(self):
        return self.now

    def advance(self, s):
        self.now += s


def make_plan(models=("M-A", "M-B", "M-C"), keys=("K1", "K2", "K3"),
              providers=("gemini", "openrouter", "grok"), **over):
    kw = dict(model_cooldown_s=300, key_cooldown_s=600, provider_cooldown_s=900,
              key_disable_s=3600, request_timeout_s=5, max_transient_retries=0,
              initial_backoff_ms=0, max_backoff_ms=0)
    kw.update(over)
    return FailoverPlan(
        providers=[
            ProviderConfig(
                name=name, priority=PRIORITY[name], enabled=True, base_url="u",
                keys=[ApiKeySlot(slot=i + 1, api_key=k) for i, k in enumerate(keys)],
                models=list(models),
            )
            for name in providers
        ],
        **kw,
    )


class Script:
    """Deterministic fake for ``adapters.generate``.

    outcomes: (provider, api_key, model) -> [("ok", text) | ("err", category)].
    Every call must be scripted: an unscripted call fails the test loudly
    (this is what proves the chain stopped / skipped as expected).
    """

    def __init__(self, outcomes):
        self.outcomes = {k: list(v) for k, v in outcomes.items()}
        self.calls = []
        self.in_flight = {}
        self.max_in_flight = {}
        self._lock = threading.Lock()

    def __call__(self, provider, api_key, model, request, **kwargs):
        rid = id(request)
        with self._lock:
            self.calls.append((provider, api_key, model))
            n = self.in_flight.get(rid, 0) + 1
            self.in_flight[rid] = n
            self.max_in_flight[rid] = max(self.max_in_flight.get(rid, 0), n)
        try:
            outs = self.outcomes.get((provider, api_key, model))
            assert outs, f"UNEXPECTED call {(provider, api_key, model)}"
            kind, val = outs.pop(0)
            if kind == "ok":
                return AIResponse(text=val, provider=provider, model=model)
            raise AIProviderError("simulated", category=val,
                                  provider=provider, model=model)
        finally:
            with self._lock:
                self.in_flight[rid] -= 1

    def max_for_request(self, request):
        return self.max_in_flight.get(id(request), 0)


@pytest.fixture
def patch_adapter(monkeypatch):
    script_holder = {}

    def install(outcomes):
        script = Script(outcomes)
        monkeypatch.setattr(adapters, "generate", script)
        script_holder["script"] = script
        return script

    return install


def fresh_health(plan, clock=None):
    return HealthManager.from_plan(plan, clock=clock or Clock())


def req(prompt="hi"):
    return AIRequest(prompt=prompt)


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------

def test_gemini_key1_primary_succeeds(patch_adapter):
    plan = make_plan()
    script = patch_adapter({("gemini", "K1", "M-A"): [("ok", "primary!")]})
    resp = generate_via_gemini(req(), plan=plan, health=fresh_health(plan),
                               sleep=lambda s: None)
    assert (resp.provider, resp.model, resp.text) == ("gemini", "M-A", "primary!")
    assert script.calls == [("gemini", "K1", "M-A")]


def test_gemini_primary_fails_fallback_succeeds(patch_adapter):
    plan = make_plan()
    script = patch_adapter({
        ("gemini", "K1", "M-A"): [("err", AIErrorCategory.MODEL_UNAVAILABLE)],
        ("gemini", "K1", "M-B"): [("ok", "fallback")]})
    resp = generate_via_gemini(req(), plan=plan, health=fresh_health(plan),
                               sleep=lambda s: None)
    assert (resp.model, resp.text) == ("M-B", "fallback")
    assert script.calls == [("gemini", "K1", "M-A"), ("gemini", "K1", "M-B")]


def test_gemini_key1_fails_key2_succeeds(patch_adapter):
    plan = make_plan()
    script = patch_adapter({
        ("gemini", "K1", "M-A"): [("err", AIErrorCategory.RATE_LIMITED)],
        ("gemini", "K2", "M-A"): [("ok", "via-k2")]})
    resp = generate_via_gemini(req(), plan=plan, health=fresh_health(plan),
                               sleep=lambda s: None)
    assert resp.text == "via-k2"
    assert script.calls == [("gemini", "K1", "M-A"), ("gemini", "K2", "M-A")]


def test_gemini_keys1_2_fail_key3_succeeds(patch_adapter):
    plan = make_plan()
    script = patch_adapter({
        ("gemini", "K1", "M-A"): [("err", AIErrorCategory.RATE_LIMITED)],
        ("gemini", "K2", "M-A"): [("err", AIErrorCategory.QUOTA_EXCEEDED)],
        ("gemini", "K3", "M-A"): [("ok", "via-k3")]})
    resp = generate_via_gemini(req(), plan=plan, health=fresh_health(plan),
                               sleep=lambda s: None)
    assert resp.text == "via-k3"
    assert script.calls == [("gemini", "K1", "M-A"),
                            ("gemini", "K2", "M-A"),
                            ("gemini", "K3", "M-A")]


def _all_fail(plan, provider, category):
    outs = {}
    for slot in (1, 2, 3):
        for model in ("M-A", "M-B", "M-C"):
            outs[(provider, f"K{slot}", model)] = [("err", category)]
    return outs


def test_all_gemini_fail_openrouter_succeeds(patch_adapter):
    plan = make_plan()
    outs = _all_fail(plan, "gemini", AIErrorCategory.TIMEOUT)
    outs[("openrouter", "K1", "M-A")] = [("ok", "via-openrouter")]
    script = patch_adapter(outs)
    resp = generate_with_failover(req(), plan=plan, health=fresh_health(plan),
                                  sleep=lambda s: None)
    assert resp.provider == "openrouter"
    assert [c for c in script.calls if c[0] == "grok"] == []
    assert script.calls[-1] == ("openrouter", "K1", "M-A")


def test_all_gemini_openrouter_fail_grok_succeeds(patch_adapter):
    plan = make_plan()
    outs = _all_fail(plan, "gemini", AIErrorCategory.TIMEOUT)
    outs.update(_all_fail(plan, "openrouter", AIErrorCategory.TIMEOUT))
    outs[("grok", "K1", "M-A")] = [("ok", "via-grok")]
    script = patch_adapter(outs)
    resp = generate_with_failover(req(), plan=plan, health=fresh_health(plan),
                                  sleep=lambda s: None)
    assert (resp.provider, resp.model) == ("grok", "M-A")
    seq = [c[0] for c in script.calls]
    assert seq == ["gemini"] * 9 + ["openrouter"] * 9 + ["grok"]


# ---------------------------------------------------------------------------
# Error-specific cases
# ---------------------------------------------------------------------------

def test_429_cools_only_that_key(patch_adapter):
    plan = make_plan()
    health = fresh_health(plan)
    script = patch_adapter({
        ("gemini", "K1", "M-A"): [("err", AIErrorCategory.RATE_LIMITED)],
        ("gemini", "K2", "M-A"): [("ok", "k2")]})
    generate_via_gemini(req(), plan=plan, health=health, sleep=lambda s: None)
    assert health.key_available("gemini", 1) is False
    assert health.key_available("gemini", 2) is True
    assert health.key_available("gemini", 3) is True
    assert health.model_available("gemini", "M-A") is True
    assert script.calls == [("gemini", "K1", "M-A"), ("gemini", "K2", "M-A")]
    assert health.key_state("gemini", 1).consecutive_failures == 1
    assert health.key_state("gemini", 1).last_error is not None


def test_model_unavailable_cools_only_that_model(patch_adapter):
    plan = make_plan()
    health = fresh_health(plan)
    script = patch_adapter({
        ("gemini", "K1", "M-A"): [("err", AIErrorCategory.MODEL_UNAVAILABLE)],
        ("gemini", "K1", "M-B"): [("ok", "b")]})
    generate_via_gemini(req(), plan=plan, health=health, sleep=lambda s: None)
    assert health.model_available("gemini", "M-A") is False
    assert health.model_available("gemini", "M-B") is True
    assert health.key_available("gemini", 1) is True
    assert health.provider_available("gemini") is True


def test_provider_outage_cools_provider_moves_on(patch_adapter):
    plan = make_plan()
    health = fresh_health(plan)
    script = patch_adapter({
        ("gemini", "K1", "M-A"): [("err", AIErrorCategory.PROVIDER_UNAVAILABLE)],
        ("openrouter", "K1", "M-A"): [("ok", "or")]})
    resp = generate_with_failover(req(), plan=plan, health=health,
                                  sleep=lambda s: None)
    assert resp.provider == "openrouter"
    assert script.calls == [("gemini", "K1", "M-A"), ("openrouter", "K1", "M-A")]
    assert health.provider_available("gemini") is False
    # Only the provider scope was cooled, not every key/model beneath it.
    assert health.key_available("gemini", 1) is True
    assert health.model_available("gemini", "M-A") is True


def test_timeout_bounded_retry_then_fallback(patch_adapter):
    plan = make_plan(max_transient_retries=2, initial_backoff_ms=50, max_backoff_ms=500)
    health = fresh_health(plan)
    delays = []
    script = patch_adapter({
        ("gemini", "K1", "M-A"): [("err", AIErrorCategory.TIMEOUT)] * 3,
        ("gemini", "K1", "M-B"): [("ok", "b")]})
    resp = generate_via_gemini(req(), plan=plan, health=health,
                               sleep=delays.append)
    assert resp.model == "M-B"
    assert delays == [0.05, 0.1]  # exponential backoff, capped
    assert script.calls == [("gemini", "K1", "M-A")] * 3 + [("gemini", "K1", "M-B")]


def test_timeout_recovers_without_cooldown_trace(patch_adapter):
    plan = make_plan(max_transient_retries=1)
    health = fresh_health(plan)
    script = patch_adapter({
        ("gemini", "K1", "M-A"): [("err", AIErrorCategory.TIMEOUT), ("ok", "recovered")]})
    resp = generate_via_gemini(req(), plan=plan, health=health,
                               sleep=lambda s: None)
    assert resp.text == "recovered"
    assert len(script.calls) == 2
    assert health.model_available("gemini", "M-A") is True
    assert health.key_available("gemini", 1) is True


def test_invalid_request_no_traversal(patch_adapter):
    plan = make_plan()
    health = fresh_health(plan)
    script = patch_adapter({
        ("gemini", "K1", "M-A"): [("err", AIErrorCategory.INVALID_REQUEST)]})
    with pytest.raises(AIProviderError) as excinfo:
        generate_with_failover(req(), plan=plan, health=health,
                               sleep=lambda s: None)
    assert excinfo.value.category is AIErrorCategory.INVALID_REQUEST
    assert script.calls == [("gemini", "K1", "M-A")]  # single attempt total
    assert health.key_available("gemini", 1) is True  # nothing cooled


def test_invalid_api_key_disabled_next_key_serves(patch_adapter):
    plan = make_plan()
    health = fresh_health(plan)
    script = patch_adapter({
        ("gemini", "K1", "M-A"): [("err", AIErrorCategory.INVALID_API_KEY)],
        ("gemini", "K2", "M-A"): [("ok", "k2")]})
    resp = generate_via_gemini(req(), plan=plan, health=health,
                               sleep=lambda s: None)
    assert resp.text == "k2"
    assert health.key_available("gemini", 1) is False
    assert health.key_state("gemini", 1).status.name == "DISABLED"
    assert health.model_available("gemini", "M-A") is True


def test_cooldown_expiry_reenables_resource():
    clock = Clock()
    plan = make_plan()
    health = fresh_health(plan, clock)
    health.record_failure("gemini", 1, "M-A",
                          AIProviderError("x", category=AIErrorCategory.RATE_LIMITED,
                                          provider="gemini", model="M-A"))
    assert health.key_available("gemini", 1) is False
    clock.advance(601)
    assert health.key_available("gemini", 1) is True
    assert health.key_state("gemini", 1).status.name == "HEALTHY"


# ---------------------------------------------------------------------------
# Critical behavior
# ---------------------------------------------------------------------------

def test_success_stops_chain_immediately(patch_adapter):
    # Only the first slot is scripted: ANY further call fails the test.
    plan = make_plan()
    script = patch_adapter({("gemini", "K1", "M-A"): [("ok", "stop")]})
    resp = generate_with_failover(req(), plan=plan, health=fresh_health(plan),
                                  sleep=lambda s: None)
    assert resp.text == "stop"
    assert script.calls == [("gemini", "K1", "M-A")]


def test_single_in_flight_call_per_request(patch_adapter):
    plan = make_plan()
    outs = _all_fail(plan, "gemini", AIErrorCategory.TIMEOUT)
    outs.update(_all_fail(plan, "openrouter", AIErrorCategory.TIMEOUT))
    outs[("grok", "K1", "M-A")] = [("ok", "end")]
    script = patch_adapter(outs)
    request = req()
    generate_with_failover(request, plan=plan, health=fresh_health(plan),
                           sleep=lambda s: None)
    assert script.max_for_request(request) == 1
    assert len(script.calls) == 19  # 9 + 9 + 1, strictly sequential


def test_cooled_scopes_are_skipped(patch_adapter):
    plan = make_plan()
    clock = Clock()
    health = fresh_health(plan, clock)
    health.mark_provider_cooldown("gemini")  # whole provider skipped
    health.mark_key_cooldown("openrouter", 1)  # key 1 skipped
    health.mark_model_cooldown("grok", "M-A")  # model A skipped
    script = patch_adapter({
        ("openrouter", "K2", "M-A"): [("ok", "or-k2")]})
    resp = generate_with_failover(req(), plan=plan, health=health,
                                  sleep=lambda s: None)
    assert (resp.provider, resp.model) == ("openrouter", "M-A")
    assert script.calls == [("openrouter", "K2", "M-A")]
    assert [c for c in script.calls if c[0] == "gemini"] == []
    assert [c for c in script.calls if c[0] == "grok" and c[2] == "M-A"] == []


def test_later_request_returns_after_expiry(patch_adapter):
    plan = make_plan()
    clock = Clock()
    health = fresh_health(plan, clock)
    outs = _all_fail(plan, "gemini", AIErrorCategory.TIMEOUT)
    outs[("openrouter", "K1", "M-A")] = [("ok", "or1")]
    script = patch_adapter(outs)
    # Request 1: gemini provider stays usable (TIMEOUT cools nothing)...
    resp = generate_with_failover(req("one"), plan=plan, health=health,
                                  sleep=lambda s: None)
    assert resp.provider == "openrouter"
    # ...so cool gemini explicitly, then expire it: request 2 returns to gemini.
    health.mark_provider_cooldown("gemini")
    assert health.provider_available("gemini") is False
    clock.advance(901)
    assert health.provider_available("gemini") is True
    script2 = patch_adapter({("gemini", "K1", "M-A"): [("ok", "back")]})
    resp2 = generate_with_failover(req("two"), plan=plan, health=health,
                                   sleep=lambda s: None)
    assert resp2.provider == "gemini"
    assert script2.calls == [("gemini", "K1", "M-A")]


class Capture(logging.Handler):
    """Direct capture on OUR logger (immune to root-handler plumbing quirks)."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_no_secret_in_logs(patch_adapter):
    from backend.ai.observability import get_ai_logger

    logger = get_ai_logger()
    cap = Capture()
    old_level = logger.level
    old_disabled = logger.disabled
    # The repo session fixture runs alembic migrations in-process, and
    # database/migrations/env.py calls logging.config.fileConfig(), which
    # disables ALL pre-existing loggers by default (alembic.ini only defines
    # root/sqlalchemy/alembic). Re-enable ours for this capture.
    logger.disabled = False
    logger.addHandler(cap)
    logger.setLevel(logging.DEBUG)
    plan = FailoverPlan(
        providers=[
            ProviderConfig(name="gemini", priority=1, enabled=True, base_url="u",
                           keys=[ApiKeySlot(slot=1, api_key=CANARY_KEY)],
                           models=["M-A"]),
            ProviderConfig(name="openrouter", priority=2, enabled=True, base_url="u",
                           keys=[ApiKeySlot(slot=1, api_key="OK1")],
                           models=["O-A"]),
        ],
        model_cooldown_s=300, key_cooldown_s=600, provider_cooldown_s=900,
        key_disable_s=3600, request_timeout_s=5, max_transient_retries=0,
        initial_backoff_ms=0, max_backoff_ms=0,
    )
    script = patch_adapter({
        ("gemini", CANARY_KEY, "M-A"): [("err", AIErrorCategory.RATE_LIMITED)],
        ("gemini", "K2", "M-A"): [("err", AIErrorCategory.TIMEOUT)],
        ("openrouter", "OK1", "O-A"): [("err", AIErrorCategory.SERVER_ERROR)]})
    try:
        with pytest.raises(AIProviderError):
            generate_with_failover(AIRequest(prompt=CANARY_PROMPT), plan=plan,
                                   health=fresh_health(plan),
                                   sleep=lambda s: None)
    finally:
        logger.removeHandler(cap)
        logger.setLevel(old_level)
        logger.disabled = old_disabled
    blob = "\n".join(cap.messages)
    assert cap.messages, "expected failover log lines"
    assert CANARY_KEY not in blob
    assert CANARY_PROMPT not in blob
    assert "Bearer" not in blob and "Authorization" not in blob
    # ...but routing signal is present: slot numbers, models, categories.
    assert "key=1" in blob and "model=M-A" in blob
    assert "rate_limited" in blob and "server_error" in blob
    assert "request_id=" in blob


def test_terminal_error_is_user_safe(patch_adapter):
    plan = make_plan(providers=("gemini",))
    outs = _all_fail(plan, "gemini", AIErrorCategory.SERVER_ERROR)
    script = patch_adapter(outs)
    with pytest.raises(AIProviderError) as excinfo:
        generate_with_failover(req(), plan=plan, health=fresh_health(plan),
                               sleep=lambda s: None)
    err = excinfo.value
    assert err.category is AIErrorCategory.PROVIDER_UNAVAILABLE
    assert err.public_message == (
        "The AI service is temporarily unavailable. Please try again shortly.")
    assert "K1" not in str(err) and "M-A" not in str(err)
    assert isinstance(err.__cause__, AIProviderError)


def test_primary_model_is_first():
    plan = make_plan(models=("PRIMARY", "F1", "F2"))
    assert plan.providers[0].models[0] == "PRIMARY"
    assert plan.providers[0].primary_model == "PRIMARY"


# ---------------------------------------------------------------------------
# Concurrency: many user requests may overlap, but each request's own
# fallback sequence must stay strictly sequential (max 1 in-flight call).
# ---------------------------------------------------------------------------

def test_concurrent_requests_each_sequential(monkeypatch):
    plan = make_plan(providers=("gemini",), keys=("K1", "K2"), models=("M-A", "M-B"))
    calls = []
    seen = set()
    in_flight = {}
    max_in_flight = {}
    lock = threading.Lock()

    def tracking_generate(provider, api_key, model, request, **kwargs):
        rid = id(request)
        with lock:
            calls.append((provider, api_key, model))
            n = in_flight.get(rid, 0) + 1
            in_flight[rid] = n
            max_in_flight[rid] = max(max_in_flight.get(rid, 0), n)
            first = rid not in seen
            seen.add(rid)
        time.sleep(0.002)  # widen the race window between threads
        with lock:
            in_flight[rid] -= 1
        if first:
            raise AIProviderError("simulated", category=AIErrorCategory.TIMEOUT,
                                  provider=provider, model=model)
        return AIResponse(text=f"ok-{rid}", provider=provider, model=model)

    monkeypatch.setattr(adapters, "generate", tracking_generate)

    requests = [req(f"user-{i}") for i in range(8)]
    results = {}
    errors = []

    def worker(request):
        try:
            # Isolated health per request so outcomes stay deterministic.
            # (TIMEOUT cools nothing anyway; isolation removes all doubt.)
            results[id(request)] = generate_with_failover(
                request, plan=plan, health=fresh_health(plan),
                sleep=lambda s: None)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(r,)) for r in requests]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:1]
    assert len(results) == 8
    for r in requests:
        # First attempt fails TIMEOUT -> next model succeeds: exactly 2 calls,
        # never overlapping within this request.
        assert max_in_flight[id(r)] == 1, f"parallel calls within one request! {max_in_flight}"
        assert results[id(r)].text == f"ok-{id(r)}"
    assert len(calls) == 16


# ---------------------------------------------------------------------------
# Robustness: startup, disabled scopes, malformed configuration
# ---------------------------------------------------------------------------

def test_missing_keys_controlled_error_zero_calls(patch_adapter):
    plan = FailoverPlan(
        providers=[
            ProviderConfig(name=name, priority=PRIORITY[name], enabled=True,
                           base_url="u",
                           keys=[ApiKeySlot(slot=i + 1, api_key="") for i in range(3)],
                           models=["M-A"])
            for name in ("gemini", "openrouter", "grok")
        ],
        model_cooldown_s=300, key_cooldown_s=600, provider_cooldown_s=900,
        key_disable_s=3600, request_timeout_s=5, max_transient_retries=0,
        initial_backoff_ms=0, max_backoff_ms=0,
    )
    script = patch_adapter({})  # all slots unconfigured: nothing callable
    with pytest.raises(AIProviderError) as excinfo:
        generate_with_failover(req(), plan=plan, health=fresh_health(plan),
                               sleep=lambda s: None)
    assert excinfo.value.category is AIErrorCategory.PROVIDER_UNAVAILABLE
    assert script.calls == []


def test_app_settings_keys_optional():
    from backend.database.config import Settings

    for var in (
        "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3",
        "OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY_3",
        "GROK_API_KEY_1", "GROK_API_KEY_2", "GROK_API_KEY_3",
    ):
        assert Settings.model_fields[var].default == "", var
    # Disabled flags and model lists are optional with sane defaults too.
    assert Settings.model_fields["GEMINI_ENABLED"].default is True
    assert Settings.model_fields["OPENROUTER_MODELS"].default != ""


def test_disabled_provider_skipped_without_crash(patch_adapter):
    plan = FailoverPlan(
        providers=[
            ProviderConfig(name="gemini", priority=1, enabled=False, base_url="u",
                           keys=[ApiKeySlot(slot=1, api_key="K1")], models=["M-A"]),
            ProviderConfig(name="openrouter", priority=2, enabled=True, base_url="u",
                           keys=[ApiKeySlot(slot=1, api_key="OK1")], models=["O-A"]),
        ],
        model_cooldown_s=300, key_cooldown_s=600, provider_cooldown_s=900,
        key_disable_s=3600, request_timeout_s=5, max_transient_retries=0,
        initial_backoff_ms=0, max_backoff_ms=0,
    )
    script = patch_adapter({("openrouter", "OK1", "O-A"): [("ok", "or")]})
    resp = generate_with_failover(req(), plan=plan, health=fresh_health(plan),
                                  sleep=lambda s: None)
    assert resp.provider == "openrouter"
    assert [c for c in script.calls if c[0] == "gemini"] == []


def test_empty_model_list_skips_provider(patch_adapter):
    plan = FailoverPlan(
        providers=[
            ProviderConfig(name="gemini", priority=1, enabled=True, base_url="u",
                           keys=[ApiKeySlot(slot=1, api_key="K1")], models=[]),
            ProviderConfig(name="grok", priority=3, enabled=True, base_url="u",
                           keys=[ApiKeySlot(slot=1, api_key="GK1")], models=["R-A"]),
        ],
        model_cooldown_s=300, key_cooldown_s=600, provider_cooldown_s=900,
        key_disable_s=3600, request_timeout_s=5, max_transient_retries=0,
        initial_backoff_ms=0, max_backoff_ms=0,
    )
    script = patch_adapter({("grok", "GK1", "R-A"): [("ok", "grok")]})
    resp = generate_with_failover(req(), plan=plan, health=fresh_health(plan),
                                  sleep=lambda s: None)
    assert resp.provider == "grok"
    assert [c for c in script.calls if c[0] == "gemini"] == []


def test_disabled_key_slot_skipped():
    plan = FailoverPlan(
        providers=[
            ProviderConfig(
                name="gemini", priority=1, enabled=True, base_url="u",
                keys=[ApiKeySlot(slot=1, api_key="K1", enabled=False),
                      ApiKeySlot(slot=2, api_key="K2")],
                models=["M-A"]),
        ],
        model_cooldown_s=300, key_cooldown_s=600, provider_cooldown_s=900,
        key_disable_s=3600, request_timeout_s=5, max_transient_retries=0,
        initial_backoff_ms=0, max_backoff_ms=0,
    )
    assert plan.providers[0].keys[0].is_configured is False
    assert [k.slot for k in plan.providers[0].keys if k.is_configured] == [2]


def test_malformed_cooldown_rejected_clearly():
    from pydantic import ValidationError

    from backend.database.config import Settings

    with pytest.raises(ValidationError) as excinfo:
        Settings(DATABASE_URL="sqlite:///./x.db", AI_KEY_COOLDOWN_S="bogus")
    assert "AI_KEY_COOLDOWN_S" in str(excinfo.value)


def test_negative_cooldown_clamped_not_fatal():
    clock = Clock()
    health = HealthManager(key_cooldown_s=-10, model_cooldown_s=-5,
                           provider_cooldown_s=-1, clock=clock)
    health.record_failure("gemini", 1, "M-A",
                          AIProviderError("x", category=AIErrorCategory.RATE_LIMITED,
                                          provider="gemini", model="M-A"))
    assert health.key_available("gemini", 1) is True


def test_blank_model_csv_falls_back_to_defaults():
    from types import SimpleNamespace

    from backend.ai.failover_config import DEFAULT_OPENROUTER_MODELS

    plan = build_failover_plan(SimpleNamespace(
        GEMINI_API_KEY_1="k", GEMINI_API_KEY_2="", GEMINI_API_KEY_3="",
        GEMINI_MODELS="gemini-2.5-flash", GEMINI_ENABLED=True, GEMINI_BASE_URL="u",
        OPENROUTER_API_KEY_1="", OPENROUTER_API_KEY_2="", OPENROUTER_API_KEY_3="",
        OPENROUTER_MODELS="   ", OPENROUTER_ENABLED=True, OPENROUTER_BASE_URL="u",
        GROK_API_KEY_1="", GROK_API_KEY_2="", GROK_API_KEY_3="",
        GROK_MODELS="g-x", GROK_ENABLED=True, GROK_BASE_URL="u",
        AI_MODEL_COOLDOWN_S=300, AI_KEY_COOLDOWN_S=600, AI_PROVIDER_COOLDOWN_S=900,
        AI_KEY_DISABLE_S=3600, AI_REQUEST_TIMEOUT_S=5, AI_MAX_TRANSIENT_RETRIES=1,
        AI_INITIAL_BACKOFF_MS=1, AI_MAX_BACKOFF_MS=2))
    by_name = {p.name: p for p in plan.providers}
    assert by_name["openrouter"].models == DEFAULT_OPENROUTER_MODELS
    assert by_name["gemini"].models == ["gemini-2.5-flash"]
    assert by_name["grok"].models == ["g-x"]


def test_unknown_provider_clear_error(patch_adapter):
    from backend.ai.cascade import cascade_provider

    plan = make_plan()
    script = patch_adapter({})
    with pytest.raises(AIProviderError) as excinfo:
        cascade_provider("nope", req(), plan=plan, health=fresh_health(plan),
                         sleep=lambda s: None)
    assert excinfo.value.category is AIErrorCategory.PROVIDER_UNAVAILABLE
    assert script.calls == []


def test_public_message_never_leaks_identifiers():
    for cat in AIErrorCategory:
        msg = AIProviderError("raw provider text 123", category=cat).public_message
        lowered = msg.lower()
        for token in ("gemini", "openrouter", "grok", "model", "key", "bearer",
                      "authorization", "http", "123"):
            assert token not in lowered, (cat, msg)
        assert cat.value not in lowered, (cat, msg)


def test_log_line_shapes():
    import re

    from backend.ai import observability as obs

    assert re.match(r"^[0-9a-f]{12}$", obs.new_request_id())
    assert obs.format_cooldowns(True, False, False, False) == "key"
    assert obs.format_cooldowns(True, True, False, True) == "key(disabled)+model"
    assert obs.format_cooldowns() == "none"
