# WanderAI-Backend

FastAPI backend for Traveller-App and Operator-Web. It owns authentication, API routes, business logic, AI integrations, external providers, SQLAlchemy models, Alembic migrations, and database seed data.

Operator authentication currently preserves the existing behavior but does not appear to enforce durable authorization on every backend request. This remains a future security task.

The `legacy-node/` directory is reference-only and is not part of normal startup.

## Validation

- `backend.main` imports successfully from this repository.
- `pytest tests -q` on a fresh database passes: **196 passed** (1 pre-existing,
  unrelated failure: a test expecting frontend files absent from this repo).
- Run the suite against an empty database with `DATABASE_URL` pointed at it;
  the shared `tests/conftest.py` fixture migrates + seeds automatically.
- The frontend repositories build and type-check independently.
- Operator authentication retains the existing behavior but does not yet enforce durable authorization on every backend request.

## Configuration

Server-side keys must stay in `.env` (see `.env.example`):

- `GEMINI_API_KEY` — Google Gemini (model cascade is `GEMINI_MODEL_FALLBACKS`
  in `backend/ai/gemini_service.py`; all AI paths degrade to grounded
  deterministic fallbacks when the key is blocked/unset — never mock data)
- `SERPAPI_API_KEY` — SerpApi (hotels/restaurants, never exposed to frontend).
  Successful hotel searches are cached 6h process-locally; quota exhaustion
  surfaces as an explicit error hint, not a bare failure
- `WEATHER_BASE_URL` / `WEATHER_API_KEY` — Weather provider (default Open-Meteo `https://api.open-meteo.com/v1/forecast`, no key required; set empty to disable → 503,TTL cache 600s)
- `PLACES_TIMEOUT_S` (default 25.0) / `PLACES_RADIUS_M` (default 30000) —
  Overpass city queries need the headroom; empty wide queries retry once at 10km
- `DATABASE_URL`, `TRAVELER_JWT_SECRET` (7-day sessions), etc.

## Run

```text
venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

> **Restart the process after every backend change** — settings and code load
> at startup. Verify the live version at `GET /api/health` and
> `GET /api/docs`. The frontend must use **one** base URL for all calls
> (`VITE_TOURFLOW_API_URL`); trip, guide, and auth calls to different
> servers/databases is the #1 cause of "not found" ghosts.

## AI Guide (persistent, context-aware, trip-scoped)

- `POST /api/guide/chat` (alias `POST /api/chat`) body `{message, tripId?}` →
  `{response, trip_id, greeting, action, suggestions, trip_card}`
- `GET /api/guide/history?tripId=` → `{trip_id, greeting, messages[], has_active_trip, trip_card}`
- `GET /api/guide/greeting?tripId=` → `{trip_id, greeting, has_active_trip, user_name}`
- Auth is traveler Bearer JWT; `user_id` is derived server-side, never trusted
  from clients. Conversations persist per (user, trip) in `guide_messages` (+
  rolling summaries); only the latest ~15 turns + summary + trip context reach
  the model. Unknown explicit IDs return 404 **with the caller's own active
  trip attached** (`active_trip`) so the UI can self-heal.
- The Guide answers grounded in real trip/catalog data (days, bookings, budget
  breakdown, hotel change via chat, honest "unavailable" for live weather) and
  never invents names, prices, or bookings. `trip_card` feeds side panels
  (budget spent/remaining, bookings count).

## Trip building (any place name works)

- `POST /api/trips` accepts catalog `destination_id` **or** free-text
  `destination_name`. Unknown places resolve via Gemini research, falling back
  to live providers (Nominatim geocode + SerpApi hotels + OSM places, marked
  `inventory_source="live"`). Transport has no live provider and is optional —
  trips simply carry no transfer items without it.
- `duration_days` is derived from `start_date`/`end_date` when both are sent
  (`end < start` → 422). Generation spreads ~2 stops/day across all days
  (pace: relaxed ≈1/day, balanced/packed ≈2/day), resolves a photo per stop,
  guarantees the planned total stays within budget, and covers empty trailing
  days with explicit leisure/check-out notes.
- Mutations return the updated trip: `PUT /trips/{id}` (dates/pace/…),
  `POST /trips/{id}/optimize` (budget-fit rebuild — call after date/pace edits),
  `change-accommodation`, `swap/add/delete/edit/toggle-activity`,
  `POST /trips/{id}/confirm` (idempotent; requires dates).
- `GET /api/trips/{id}/map` → `{destination, center, days[] with plotted
  stops (real lat/lng), unmapped_count}` for Leaflet/OSM rendering (no key).
- Trip creation uses the session owner when a Bearer token is present, so new
  trips belong to the logged-in traveler (the Guide can see them).

## Traveler auth, profile & avatar

- `POST /api/auth/traveler/signup {full_name, email, password(8–72)}` →
  `201 {user, token}` (409 duplicate, 422 validation); `POST
  /api/auth/traveler/login {email, password}` → `200 {user, token}`;
  `GET /api/auth/traveler/me` → `{id, email, full_name}`. **Token key is
  exactly `token`.**
- `GET/PUT/PATCH /api/traveler/profile`, `GET/PUT /api/traveler/preferences`,
  `GET /api/traveler/trips` (own trips only — use for pickers, never the
  unauthenticated trip list).
- `POST /api/traveler/avatar` (multipart `file`, image/*, ≤5 MB, Bearer) stores
  DB-backed bytes; `GET /api/traveler/avatar/{user_id}` is public (fallback to
  initial on 404); `DELETE /api/traveler/avatar` resets. Profile responses
  carry `has_avatar`.

## Voice / preference extraction

- `POST /api/ai/extract-preferences` accepts `{text_prompt}` **or** the voice
  alias `{text}` (+ optional `context`) → `{detected_destination,
  budget_tier, budget_amount, budget_currency, interests, travel_companions,
  traveler_count, duration_days, pace, special_requests, source}`.
- Destination matching uses live catalog names; the rule-based extractor runs
  both when Gemini is unconfigured **and** when it fails, so a blocked key
  never returns emptier results than no key.

## Seed data & migrations

- Catalog destinations with full inventory (≥8 activities, hotels, transport,
  real coordinates): Manali, Goa, Kashmir, Kerala, Rajasthan, Udaipur.
- `run_seed()` is idempotent and backfills older databases on startup
  (inventory rows, coordinates, catalog repairs).
- Alembic single-head chain through `0012_avatar` (`guide_messages`,
  summaries, traveler auth, ops tables, avatar columns). Never create a
  second head — chain new revisions onto the current head.
 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CLIENT LAYER (React 19)                                │
│  ┌─────────────────────────────┐                    ┌────────────────────────────────┐  │
│  │     Traveler Workspace      │                    │    Operator Enterprise Suite   │  │
│  │ (Hero, Planner, Itinerary,  │                    │ (Dashboard, Fleet, Replan,     │  │
│  │  Map, PDF, Concierge Chat)  │                    │  Vendors, Analytics, AI Ops)   │  │
│  └──────────────┬──────────────┘                    └───────────────┬────────────────┘  │
│                 │                                                   │                   │
│                 └─────────────────────┬─────────────────────────────┘                   │
│                                       ▼                                                 │
│                     Zustand Store / Unified API Client (`/src/services/api.ts`)         │
└───────────────────────────────────────┬─────────────────────────────────────────────────┘
                                        │ HTTP / JSON (Port 3000 / 8000)
┌───────────────────────────────────────▼─────────────────────────────────────────────────┐
│                                  BACKEND & API LAYER                                    │
│  ┌────────────────────────────────────────┐  ┌───────────────────────────────────────┐  │
│  │         Node/Express Gateway           │  │         Python FastAPI Core           │  │
│  │             (server.ts)                │  │          (backend/main.py)            │  │
│  ├────────────────────────────────────────┤  ├───────────────────────────────────────┤  │
│  │ • API Routes (/api/*)                  │  │ • REST API Endpoints (/api/*)         │  │
│  │ • Vite SSR/SPA Middleware              │  │ • SQLAlchemy 2.0 ORM Engine           │  │
│  │ • AI Service Bridges                   │  │ • Pydantic v2 Validation Schemas      │  │
│  │ • Dynamic Transport Route Synthesizer  │  │ • Recommendation & Replanning Engines │  │
│  └───────────────────┬────────────────────┘  └───────────────────┬───────────────────┘  │
│                      │                                           │                      │
│                      └─────────────────────┬─────────────────────┘                      │
└────────────────────────────────────────────┼────────────────────────────────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
┌───────────────────────────────────────────┐ ┌───────────────────────────────────────────┐
│           AI INTELLIGENCE LAYER           │ │           PERSISTENCE LAYER               │
│  ┌─────────────────────────────────────┐  │ │  ┌─────────────────────────────────────┐  │
│  │   Google Gemini 2.5/3.7 Models      │  │ │  │   PostgreSQL / SQLite Database      │  │
│  │      (`@google/genai` SDK)          │  │ │  │         (tourflow.db)               │  │
│  ├─────────────────────────────────────┤  │ │  ├─────────────────────────────────────┤  │
│  │ • Structured JSON Schema Extraction │  │ │  │ • Single Source of Truth            │  │
│  │ • Knowledge Base Grounding          │  │ │  │ • Canonical `trips` & `itinerary`   │  │
│  │ • Multi-Candidate Replan Generation │  │ │  │ • Alembic Database Migrations       │  │
│  │ • Conversational Chat Concierge     │  │ │  │ • Deterministic Seed Data           │  │
│  └─────────────────────────────────────┘  │ │  └─────────────────────────────────────┘  │
└───────────────────────────────────────────┘ └───────────────────────────────────────────┘
```

### Core Architecture Subsystems
1. **Frontend**: React 19, TypeScript, Tailwind CSS v4, Lucide Icons, Motion animations, Leaflet map renderer, jsPDF client exporter.
2. **Backend Services**: Express TypeScript server (`server.ts`) and Python FastAPI engine (`backend/main.py`) exposing unified `/api` REST contracts.
3. **Database**: PostgreSQL (with SQLite zero-config fallback for testing/local agility) accessed via SQLAlchemy ORM models and Alembic versioned migrations.
4. **AI Integration**: Centralized `GeminiService` using `@google/genai` with automatic fallback cascade (`GEMINI_MODEL_FALLBACKS` in `backend/ai/gemini_service.py`, currently 3.6-flash generation). All AI paths degrade to grounded deterministic answers when the key is blocked/unset.
5. **Interactive Mapping**: Leaflet with OpenStreetMap tiles, custom marker overlays (`divIcon`), and coordinates computed via `geoCoordinates.ts`.
6. **Images**: Curated Unsplash photo catalog with robust keyword-based image resolution fallback (`SmartImage.tsx` & `imageCatalog.ts`).
7. **PDF Generation**: Native client-side PDF document compiler in `src/utils/pdfExport.ts` producing A4 structured printouts with financial breakdowns and daily time slots.
8. **Real-Time Synchronization**: Traveler and Operator access the **exact same canonical records**. Changes made by either side persist immediately to the shared database and reflect across both portals.

# Recent Changes 
 Made the AI Guide persistent and context-aware. It now knows the logged-in user by name, plus their active trip, itinerary, bookings, budget, and full conversation history. Nothing is lost on refresh, and one traveler's data never leaks to another. Trip creation was opened up to any place name: seeded 6 catalog destinations with full inventory, with unknown names resolving via Gemini research and — when that fails — via live providers (SerpApi hotels + OSM places).

Fixed the itinerary engine: duration derived from dates, ~2 stops/day spread proportionally across all days, a photo on every stop, planning guaranteed within budget (caught and fixed a real blowout: ₹76,950 planned on a ₹50,000 budget), leisure notes for empty trailing days, and a map-ready endpoint. The Guide got a conversational brain: greetings, budget breakdowns, catalog Q&A, hotel changes via chat, and honest "unavailable" answers where data doesn't exist. Fixed trip ownership (new trips belong to the logged-in user), session restore, and SerpApi quota protection (caching + explicit hints).

Profile track: login/signup contract, profile page on real APIs, DB-backed avatar upload, and a voice-input extract contract whose rule-based fallback is so strong it pulls Udaipur · 6 days · 2 travelers · ₹75,000 out of a sentence even with the Gemini key blocked. Home page got proper Sign In / Create Account buttons.

