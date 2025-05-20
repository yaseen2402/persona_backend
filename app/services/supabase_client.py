# app/services/supabase_client.py
from supabase import create_client, Client
from app.core.config import SUPABASE_URL, SUPABASE_SERVICE_KEY
import sys

_supabase_client_instance: Client | None = None

def get_supabase_client() -> Client | None:
    global _supabase_client_instance
    if _supabase_client_instance is None:
        if SUPABASE_URL and SUPABASE_SERVICE_KEY:
            try:
                _supabase_client_instance = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
                print("Supabase client initialized successfully (shared instance).")
                sys.stdout.flush()
            except Exception as e:
                print(f"Error initializing Supabase client (shared instance): {e}")
                sys.stdout.flush()
                _supabase_client_instance = None # Ensure it's None on failure
        else:
            print("SUPABASE_URL or SUPABASE_SERVICE_KEY not set. Cannot initialize shared Supabase client.")
            sys.stdout.flush()
    return _supabase_client_instance

# Initialize on first import if desired, or let services call the getter.
# Calling it here makes it available as `from app.services.supabase_client import supabase`
supabase: Client | None = get_supabase_client()

# Helper for Celery tasks to get a NEW client instance
def get_new_supabase_client_for_task() -> Client | None:
    # This function is specifically for Celery tasks to get a fresh client.
    task_supabase_url = SUPABASE_URL # From config
    task_supabase_service_key = SUPABASE_SERVICE_KEY # From config
    
    if not task_supabase_url or not task_supabase_service_key:
        print("[Celery Task/Supabase Init] SUPABASE_URL or SUPABASE_SERVICE_KEY missing for worker task.")
        sys.stdout.flush()
        return None
    try:
        client = create_client(task_supabase_url, task_supabase_service_key)
        print("[Celery Task/Supabase Init] New Supabase client initialized successfully for worker task.")
        sys.stdout.flush()
        return client
    except Exception as e:
        print(f"[Celery Task/Supabase Init] Error initializing new Supabase client for worker task: {e}")
        sys.stdout.flush()
        return None