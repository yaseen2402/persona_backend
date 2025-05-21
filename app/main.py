from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import JSONResponse
import traceback
import sys
import os



print("Initializing configuration in main.py...")
sys.stdout.flush()
from app.core import config 

print("Configuration loaded. Initializing services...")
sys.stdout.flush()
from app.services import supabase_client, llm_service, embedding_service, persona_service, user_service

print("Services initialized. Importing API routers...")
sys.stdout.flush()
from app.api import chat, users, messages, models


app = FastAPI(
    title="AI Chat API",
    description="API for interacting with AI personas and managing user data.",
    version="1.0.0"
)

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


API_PREFIX = "" 

app.include_router(chat.router, prefix=API_PREFIX, tags=["Chat"])
app.include_router(users.router, prefix=API_PREFIX, tags=["User Profile & Stats"])
app.include_router(messages.router, prefix=API_PREFIX, tags=["Chat Messages"])
app.include_router(models.router, prefix=API_PREFIX, tags=["AI Models/Personas"])

@app.get("/", tags=["Root"])
async def read_root():
    print("Root path '/' accessed.")
    sys.stdout.flush()
    return {"message": "Welcome to the AI Chat API. See /docs for API details."}


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI app directly with Uvicorn (for local dev without Docker Compose)...")
    sys.stdout.flush()
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True, log_level="debug")