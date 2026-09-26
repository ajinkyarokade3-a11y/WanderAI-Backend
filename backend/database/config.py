from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load the repo-root .env into the process environment (no override of real
# env vars) so BOTH pydantic-settings fields AND direct os.getenv() reads
# (e.g. OPERATOR_LOGIN_PASSWORD in backend/api/routes.py::operator_login)
# observe the same values regardless of the caller's working directory.
# Path is anchored to this file so `pytest tests/...`, `uvicorn`, and IDE
# runs from any cwd resolve the same file.
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    PROJECT_NAME: str = "TourFlow AI"
    API_V1_STR: str = "/api"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "info"
    BACKEND_PORT: int = 8000
    DATABASE_URL: str
    GEMINI_API_KEY: str = ""
    # ------------------------------------------------------------------
    # AI failover hierarchy: Provider -> API Key -> Model (sequential).
    # Provider priority is fixed: Gemini (1) -> OpenRouter (2) -> Grok (3).
    # No secrets are hardcoded; every credential comes from the environment.
    # Model lists are CSV strings so they can change without touching code.
    # ------------------------------------------------------------------
    # Gemini (priority 1). GEMINI_API_KEY is kept as a legacy alias for key 1.
    GEMINI_API_KEY_1: str = ""
    GEMINI_API_KEY_2: str = ""
    GEMINI_API_KEY_3: str = ""
    GEMINI_MODELS: str = "gemini-2.5-flash,gemini-2.5-flash-lite,gemini-3.1-flash-lite,gemini-3.5-flash-lite,gemini-3.5-flash"
    GEMINI_ENABLED: bool = True
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com"
    # OpenRouter (priority 2, OpenAI-compatible). IDs verified Sep 2026 from
    # https://openrouter.ai/collections/free-models and the live :free catalog;
    # the free roster churns, so override via env without code changes.
    OPENROUTER_API_KEY_1: str = ""
    OPENROUTER_API_KEY_2: str = ""
    OPENROUTER_API_KEY_3: str = ""
    OPENROUTER_MODELS: str = "openai/gpt-oss-20b:free,google/gemma-4-31b-it:free,nvidia/nemotron-3-nano-30b-a3b:free,cohere/north-mini-code:free"
    OPENROUTER_ENABLED: bool = True
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    # Grok / xAI (priority 3, OpenAI-compatible Chat Completions, which xAI
    # still serves alongside the newer Responses API). xAI has no $0 tier;
    # defaults are documented IDs from https://docs.x.ai/developers/models
    # (Sep 2026); primary grok-4.3 is on the standard-API pricing page.
    GROK_API_KEY_1: str = ""
    GROK_API_KEY_2: str = ""
    GROK_API_KEY_3: str = ""
    GROK_MODELS: str = "grok-4.3,grok-code-fast-1,grok-4-1-fast-reasoning"
    GROK_ENABLED: bool = True
    GROK_BASE_URL: str = "https://api.x.ai/v1"
    # Shared failover resilience tuning (router reads these; no logic here).
    AI_MODEL_COOLDOWN_S: float = 300.0
    AI_KEY_COOLDOWN_S: float = 600.0
    AI_PROVIDER_COOLDOWN_S: float = 900.0
    # Long-window skip for provider-rejected keys (INVALID_API_KEY). Still a
    # cooldown, never permanent: expiry re-enables the key automatically.
    AI_KEY_DISABLE_S: float = 3600.0
    AI_REQUEST_TIMEOUT_S: float = 30.0
    AI_MAX_TRANSIENT_RETRIES: int = 1
    AI_INITIAL_BACKOFF_MS: int = 1000
    AI_MAX_BACKOFF_MS: int = 10000
    # Live hotel search (SerpApi Google Hotels, backend-only). Empty key
    # disables live search; endpoints then report provider-unavailable.
    SERPAPI_API_KEY: str = ""
    SERPAPI_BASE_URL: str = "https://serpapi.com"
    SERPAPI_TIMEOUT_S: float = 20.0
    SERPAPI_MAX_RESULTS: int = 10
    # Live restaurant search (SerpApi Google Maps, backend-only). Reuses
    # SERPAPI_API_KEY; empty key disables live search. Optional tuning only.
    SERPAPI_RESTAURANT_MAX_RESULTS: int = 8
    # Live places/attractions (keyless providers, backend-only).
    NOMINATIM_API_URL: str = "https://nominatim.openstreetmap.org"
    OVERPASS_API_URL: str = "https://overpass-api.de/api/interpreter"
    COMMONS_API_URL: str = "https://commons.wikimedia.org/w/api.php"
    # Overpass city queries routinely need >8s; the pipeline retries tight
    # on empty wide queries, so metros answer instead of timing out blank.
    PLACES_TIMEOUT_S: float = 25.0
    PLACES_RADIUS_M: int = 30000
    PLACES_MAX_RESULTS: int = 12
    # Weather / live conditions (server-side, never exposed to frontend)
    WEATHER_BASE_URL: str = "https://api.open-meteo.com/v1/forecast"
    WEATHER_API_KEY: str = ""
    WEATHER_TIMEOUT_S: float = 8.0
    WEATHER_CACHE_TTL_S: int = 600
    WEATHER_DAYS_DEFAULT: int = 5
    # Traveler password login (JWT). Override TRAVELER_JWT_SECRET in production;
    # the built-in default is development-only.
    TRAVELER_JWT_SECRET: str = "dev-only-traveler-jwt-secret-change-in-production"
    TRAVELER_JWT_EXPIRY_DAYS: int = 7
    # Operator shared-password login. Server-side only, never frontend.
    # POST /api/auth/operator-login fail-closes with 503 while this is empty.
    # Empty default here = "not configured"; set it in the local .env only.
    OPERATOR_LOGIN_PASSWORD: str = ""


settings = Settings()


def resolve_db_url(url: str) -> str:
    """Normalize a DATABASE_URL to the psycopg (v3) SQLAlchemy dialect.

    Bare ``postgres://`` / ``postgresql://`` schemes default to the psycopg2
    driver, which is not installed — rewrite them to ``postgresql+psycopg``.
    Explicitly qualified URLs (``postgresql+psycopg://``, ``sqlite:///...``)
    pass through untouched.
    """
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    return url

