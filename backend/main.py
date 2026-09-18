import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.database.connection import engine, Base
from backend.database.config import settings
from backend.api.routes import router as api_router
from database.seed_data.seed import run_seed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tourflow_backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure DB tables exist and seed initial data if empty
    logger.info("Initializing TourFlow AI Database Tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
        # Auto seed if database is brand new
        run_seed()
    except Exception as e:
        logger.warning(f"Database initialization note: {e}")
    yield
    # Shutdown
    logger.info("TourFlow AI Backend shutting down.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="TourFlow AI - AI-Powered Personalized & Dynamic Travel Planning Platform API",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Frontend communication
ALLOWED_ORIGINS = ["*", "http://localhost:3001"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix="/api")

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # FastAPI routes Exception/500 handlers to ServerErrorMiddleware, which runs
    # OUTSIDE CORSMiddleware. Without explicit headers the live server returns a
    # plain 500 with no Access-Control-Allow-Origin, which browsers report as a
    # CORS failure (e.g. GET /api/possible-options). Mirror the CORS headers here.
    logger.exception(f"Unhandled error on {request.url.path}: {exc}")
    origin = request.headers.get("origin")
    cors_headers = {}
    if origin and ("*" in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS):
        cors_headers["Access-Control-Allow-Origin"] = origin
        cors_headers["Access-Control-Allow-Credentials"] = "true"
        cors_headers["Vary"] = "Origin"
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
        headers=cors_headers,
    )

@app.get("/")
def root():
    return {
        "app": "TourFlow AI",
        "description": "AI-Powered Personalized & Dynamic Travel Planning Platform",
        "docs": "/docs",
        "api": "/api/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
