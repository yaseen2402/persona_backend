# app/main.py
from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import JSONResponse
import traceback
import sys
import os

# Set PYTHONPATH if running locally and it's not finding 'app' package.
# Usually not needed when running with Docker if WORKDIR and commands are correct.
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir) # This is /app folder
# if project_root not in sys.path:
#    sys.path.insert(0, project_root)
# if os.path.dirname(project_root) not in sys.path: # To find 'app' from project_root
#    sys.path.insert(0, os.path.dirname(project_root))


# IMPORTANT: Initialize config first to load .env before other modules are imported
# which might depend on those environment variables.
print("Initializing configuration in main.py...")
sys.stdout.flush()
from app.core import config # This will execute config.py and load .env

print("Configuration loaded. Initializing services...")
sys.stdout.flush()
# These imports will initialize the shared clients using the loaded config
from app.services import supabase_client, llm_service, embedding_service, persona_service, user_service

print("Services initialized. Importing API routers...")
sys.stdout.flush()
from app.api import chat, users, messages, models

# This import is mainly to ensure Celery app is known if not explicitly run via celery CLI with -A
# from app.celery_app import celery_app as app_celery # Alias to avoid name clash

app = FastAPI(
    title="AI Chat API",
    description="API for interacting with AI personas and managing user data.",
    version="1.0.0"
)

# Generic Exception Handler
@app.exception_handler(Exception)
async def generic_app_exception_handler(request: FastAPIRequest, exc: Exception):
    print(f"Unhandled application exception: {exc}")
    sys.stdout.flush()
    traceback.print_exc()
    sys.stdout.flush()
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": "An unexpected error occurred."},
    )

# Include API routers
# Using a prefix like /api/v1 is common for versioning.
# If your frontend expects routes at the root, remove the prefix.
API_PREFIX = "" # Or set to "" for root paths

app.include_router(chat.router, prefix=API_PREFIX, tags=["Chat"])
app.include_router(users.router, prefix=API_PREFIX, tags=["User Profile & Stats"])
app.include_router(messages.router, prefix=API_PREFIX, tags=["Chat Messages"])
app.include_router(models.router, prefix=API_PREFIX, tags=["AI Models/Personas"])

@app.get("/", tags=["Root"])
async def read_root():
    print("Root path '/' accessed.")
    sys.stdout.flush()
    return {"message": "Welcome to the AI Chat API. See /docs for API details."}

# Optional: Uvicorn startup for direct execution (python app/main.py)
# Docker Compose will use gunicorn to run this app.main:app instance.
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI app directly with Uvicorn (for local dev without Docker Compose)...")
    sys.stdout.flush()
    # When running main.py directly from project_root: uvicorn app.main:app
    # If running from inside app directory: uvicorn main:app
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True, log_level="debug")