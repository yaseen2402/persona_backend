# app/core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent.parent

dotenv_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=dotenv_path)

print(f"Loading .env from: {dotenv_path}")

REDIS_URL = os.getenv('REDIS_URL_UPSTASH', 'rediss://localhost:6379/0') 

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_HOME = os.getenv("HF_HOME", str(BASE_DIR / ".cache/huggingface")) 

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

PERSONAS_PATH = BASE_DIR / "personas.json"

USER_MESSAGE_STATS_TABLE = 'user_message_stats'
USER_PERSONA_STATES_TABLE = 'user_persona_states'
SUBSCRIPTION_PLAN_TABLE = 'subscription_plans'
USER_SUBSCRIPTIONS_TABLE = 'user_subscriptions'
MESSAGES_TABLE = 'messages'

print(f"  REDIS_URL: {'*' * 5 + REDIS_URL[-5:] if REDIS_URL else 'Not set'}")
print(f"  GOOGLE_API_KEY: {'Set' if GOOGLE_API_KEY else 'Not set'}")
print(f"  SUPABASE_URL: {SUPABASE_URL}")
print(f"  HF_HOME: {HF_HOME}")

if not REDIS_URL:
    print("CRITICAL WARNING: REDIS_URL (or REDIS_URL_UPSTASH) is not set in .env")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    print("CRITICAL WARNING: SUPABASE_URL or SUPABASE_SERVICE_KEY is not set in .env")