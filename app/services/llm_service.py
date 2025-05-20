# app/services/llm_service.py
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import GOOGLE_API_KEY
import sys

_llm_client_instance: ChatGoogleGenerativeAI | None = None

def get_llm_client() -> ChatGoogleGenerativeAI | None:
    global _llm_client_instance
    if _llm_client_instance is None:
        if GOOGLE_API_KEY:
            try:
                # Ensure model name matches what's available and intended for your key/project
                _llm_client_instance = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
                print("Gemini model initialized successfully (shared instance).")
                sys.stdout.flush()
            except Exception as e:
                print(f"Error initializing Gemini model (shared instance): {e}")
                sys.stdout.flush()
                _llm_client_instance = None
        else:
            print("GOOGLE_API_KEY not set. Cannot initialize shared LLM client.")
            sys.stdout.flush()
    return _llm_client_instance

llm: ChatGoogleGenerativeAI | None = get_llm_client()

# Helper for Celery tasks
def get_new_llm_client_for_task() -> ChatGoogleGenerativeAI | None:
    task_google_api_key = GOOGLE_API_KEY # From config
    if not task_google_api_key:
        print("[Celery Task/LLM Init] GOOGLE_API_KEY missing for worker task.")
        sys.stdout.flush()
        return None
    try:
        llm_instance = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=task_google_api_key)
        print("[Celery Task/LLM Init] New Gemini model initialized successfully for worker task.")
        sys.stdout.flush()
        return llm_instance
    except Exception as e:
        print(f"[Celery Task/LLM Init] Error initializing new Gemini model for worker task: {e}")
        sys.stdout.flush()
        return None