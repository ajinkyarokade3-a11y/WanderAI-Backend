# WanderAI-Backend

FastAPI backend for Traveller-App and Operator-Web. It owns authentication, API routes, business logic, AI integrations, external providers, SQLAlchemy models, Alembic migrations, and database seed data.

Operator authentication currently preserves the existing behavior but does not appear to enforce durable authorization on every backend request. This remains a future security task.

The `legacy-node/` directory is reference-only and is not part of normal startup.

## Validation

- `backend.main` imports successfully from this repository.
- `pytest tests -q` passes: 95 tests passed.
- The frontend repositories build and type-check independently.
- Operator authentication retains the existing behavior but does not yet enforce durable authorization on every backend request.

## Run

```text
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
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
4. **AI Integration**: Centralized `GeminiService` using `@google/genai` with automatic fallback cascade (`gemini-2.5-flash` $\rightarrow$ `gemini-3.7-flash` $\rightarrow$ `gemini-flash-latest` $\rightarrow$ `gemini-3.1-flash-lite`).
5. **Interactive Mapping**: Leaflet with OpenStreetMap tiles, custom marker overlays (`divIcon`), and coordinates computed via `geoCoordinates.ts`.
6. **Images**: Curated Unsplash photo catalog with robust keyword-based image resolution fallback (`SmartImage.tsx` & `imageCatalog.ts`).
7. **PDF Generation**: Native client-side PDF document compiler in `src/utils/pdfExport.ts` producing A4 structured printouts with financial breakdowns and daily time slots.
8. **Real-Time Synchronization**: Traveler and Operator access the **exact same canonical records**. Changes made by either side persist immediately to the shared database and reflect across both portals.
