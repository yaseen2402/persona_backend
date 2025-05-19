import os
import jwt
import traceback
import sys # Import sys for flushing

import json
from pathlib import Path
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Header, Request as FastAPIRequest
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field # For request/response models
from typing import List, Dict, Optional, Any

import uvicorn

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

from celery import Celery

CELERY_BROKER_URL = os.getenv('REDIS_URL', 'rediss://localhost:6379/0')

celery_app = Celery(
    __name__,
    broker=CELERY_BROKER_URL,
)
app = FastAPI() 

# --- Pydantic Models for FastAPI (examples) ---
class ChatRequest(BaseModel):
    ai_id: str
    prompt: str
    history: Optional[List[Dict[str, Any]]] = []

class ChatResponse(BaseModel):
    response: str

class MessageLimitErrorResponse(BaseModel):
    error: str

class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    avatar_initial: str
    profile_pic_url: str

class Message(BaseModel):
    id: Any # Or int, or str, depending on DB
    user_id: str
    ai_id: str
    content: str
    created_at: datetime
    is_from_user: bool

class UserProfileStatsResponse(BaseModel):
    plan_type: str
    daily_message_count: int
    daily_message_limit: Optional[int]
    total_message_count: int
    total_message_limit: Optional[int]
    last_reset_at: Optional[datetime]


# Load environment variables from .env file
load_dotenv()


embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def get_supabase_client_for_task():
    # This function should NOT rely on the global 'supabase' variable.
    # It creates a NEW client.
    task_supabase_url = os.getenv("SUPABASE_URL")
    task_supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not task_supabase_url or not task_supabase_service_key:
        print("[Celery Task/Supabase Init] SUPABASE_URL or SUPABASE_SERVICE_KEY missing for worker task.")
        sys.stdout.flush()
        return None
    try:
        client = create_client(task_supabase_url, task_supabase_service_key)
        print("[Celery Task/Supabase Init] Supabase client initialized successfully for worker task.")
        sys.stdout.flush()
        return client
    except Exception as e:
        print(f"[Celery Task/Supabase Init] Error initializing Supabase client for worker task: {e}")
        sys.stdout.flush()
        return None

def get_llm_for_task():
    # This function should NOT rely on the global 'llm' variable.
    # It creates a NEW client.
    task_google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not task_google_api_key:
        print("[Celery Task/LLM Init] GOOGLE_API_KEY missing for worker task.")
        sys.stdout.flush()
        return None
    try:
        # Ensure model name matches what's available and intended
        llm_instance = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=task_google_api_key) 
        print("[Celery Task/LLM Init] Gemini model initialized successfully for worker task.")
        sys.stdout.flush()
        return llm_instance
    except Exception as e:
        print(f"[Celery Task/LLM Init] Error initializing Gemini model for worker task: {e}")
        sys.stdout.flush()
        return None
    
def get_embedding(text: str) -> list[float]:
    # Returns a normalized embedding list
    emb = embed_model.encode(text, normalize_embeddings=True)
    return emb.tolist()

def check_message_limits(user_id):
    print(f"[check_message_limits] Checking message limits for user {user_id}.")
    
    if supabase is None:
        print("[check_message_limits] Supabase is not initialized.")
        return None, None
    
    try:
        # Get today's date for checking daily message limit
        today = datetime.today().date()
        
        # Check user's message stats for today and total
        response = supabase.table('user_message_stats') \
            .select('daily_count, total_count') \
            .eq('user_id', user_id) \
            .single() \
            .execute()
        
        if not response.data:
            print(f"[check_message_limits] No data found for user {user_id}.")
            return None, None

        stats = response.data
        free_messages_today = stats['daily_count']
        total_messages = stats['total_count']
        
        # Check if user has remaining free messages for today
        if free_messages_today >= 20:
            print(f"[check_message_limits] User {user_id} has exceeded the daily free messages limit.")
            return False, "Daily free message limit exceeded."
        
        # Check if total message count exceeds 500
        if total_messages >= 500:
            print(f"[check_message_limits] User {user_id} has exceeded the total message limit.")
            return False, "Total message limit exceeded."

        return True, {'free_messages_today': free_messages_today, 'total_messages': total_messages}
        
    except Exception as e:
        print(f"[check_message_limits] Error checking message limits for user {user_id}: {e}", e)
        return False, "An error occurred while checking your message limits."
    

# Get the Google API Key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")


supabase_url = os.getenv("SUPABASE_URL") # Get Supabase URL
supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY") # Get Supabase Service Role Key
supabase_jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
PERSONAS_PATH = Path("personas.json")

SUBSCRIPTION_PLAN_TABLE = 'subscription_plans'
USER_MESSAGE_STATS_TABLE = 'user_message_stats'
USER_SUBSCRIPTIONS_TABLE = 'user_subscriptions'
USER_PERSONA_STATES_TABLE = 'user_persona_states'

with open(PERSONAS_PATH, "r", encoding="utf-8") as f:
    persona_list = json.load(f)
    AI_PERSONAS = {p["id"]: p for p in persona_list}


# --- Initialize Supabase Client (Backend) ---
supabase = None
if not supabase_url or not supabase_service_key:
    print("SUPABASE_URL or SUPABASE_SERVICE_KEY environment variable not set.")
    print("Database interaction features will be disabled.")
else:
    try:
        # Use the service role key for backend operations (bypasses RLS)
        supabase = create_client(supabase_url, supabase_service_key)
        print("Supabase client initialized successfully on the backend.")
    except Exception as e:
        supabase = None
        print(f"Error initializing Supabase client on backend: {e}", e)
        print("Database interaction features will be disabled.")

llm = None  # type: ChatGoogleGenerativeAI | None

if not google_api_key:
    print("GOOGLE_API_KEY environment variable not set.")
    print("AI chat functionality will be disabled.")
else:
    # Initialize the Langchain ChatGoogleGenerativeAI model
    # Specify the model name
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
        print("Gemini model initialized successfully.")
        # Optional: Test a simple invoke to check if key/model works
        # try:
        #     test_response = llm.invoke("Hello, test.")
        #     print(f"Gemini test invoke successful. Response preview: {test_response.content[:50]}...")
        # except Exception as test_e:
        #      print(f"Gemini test invoke failed: {test_e}. API key or model might have issues despite initialization.")

    except Exception as e:
        llm = None
        print(f"Error initializing Gemini model: {e}", e)
        print("AI chat functionality will be disabled.")


@app.exception_handler(Exception) # Generic exception handler
async def generic_exception_handler(request: FastAPIRequest, exc: Exception):
    print(f"Caught unhandled FastAPI exception: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )

@celery_app.task
def perform_analysis_task(user_id_str, ai_id, user_prompt):
    """
    Performs the AI analysis and updates the user persona state.
    This function runs asynchronously in a Celery worker.
    """
    print(f"[Celery Task] Starting analysis for user {user_id_str}, AI {ai_id}")

    # Re-initialize or access necessary components if not global/contextual
    # For this example, assuming supabase and llm are accessible globally as in app.py
    # If running workers in a different process/environment, you might need
    # to handle initialization here or pass necessary config.
    supabase = get_supabase_client_for_task()
    llm = get_llm_for_task()
    if supabase is None:
        print("[Celery Task] Supabase is not initialized.")
        # Handle this error - maybe retry or log failure
        return {"status": "error", "message": "Database not available"}
    if llm is None:
        print("[Celery Task] LLM is not initialized.")
        # Handle this error
        return {"status": "error", "message": "AI model not available"}

    try:
        # --- Fetch Current User Persona State (Logic from /analysis route) ---
        user_persona_state = None
        try:
            print(f"[Celery Task] Fetching current user persona state for user {user_id_str} and persona {ai_id}.")
            state_response = supabase.table(USER_PERSONA_STATES_TABLE) \
                .select('*') \
                .eq('user_id', user_id_str) \
                .eq('persona_id', ai_id) \
                .single() \
                .execute()

            if state_response.data:
                user_persona_state = state_response.data
                print(f"[Celery Task] Successfully fetched current user persona state.")
            else:
                print(f"[Celery Task] No existing user persona state found. Using default.")
                user_persona_state = {
                    'attachment_level': 1, 'trust_level': 1, 'mood': 'neutral',
                    'introversion_level': 3, 'relationship_stage': 'stranger',
                    'openness': 3, 'energy_level': 3, 'user_summary': '', 'user_facts': {}
                }

        except Exception as e:
            print(f"[Celery Task] Error fetching current user persona state: {e}")
            user_persona_state = {
                 'attachment_level': 1, 'trust_level': 1, 'mood': 'neutral',
                 'introversion_level': 3, 'relationship_stage': 'stranger',
                 'openness': 3, 'energy_level': 3, 'user_summary': '', 'user_facts': {}
            }
            print(f"[Celery Task] Using default state due to fetch error.")


        # --- Prepare Prompt for LLM Analysis (Logic from /analysis route) ---
        analysis_prompt_text = f"""
        Analyze the following user prompt in the context of the user's current relationship and state with the AI persona (ID: {ai_id}).
        Based on the user's prompt, determine if any of the following attributes of the user's state should be updated.
        Provide the suggested updates in JSON format. Only include the attributes that need to be updated.
        For numerical attributes (attachment_level, trust_level, introversion_level, openness, energy_level), suggest an increment (+1), a decrement (-1), or a specific value (1-10).
        For text attributes (mood, relationship_stage, user_summary), provide the new value.
        For user_facts (jsonb), provide a JSON object with key-value pairs to add or update.

        Current User State:
        {json.dumps(user_persona_state, indent=2)}

        User Prompt: "{user_prompt}"

        Suggested Updates (JSON format, only include attributes to update):
        """

        print(f"[Celery Task] Sending analysis prompt to LLM.")

        # --- Invoke LLM for Analysis (Logic from /analysis route) ---
        analysis_llm_response = llm.invoke([HumanMessage(content=analysis_prompt_text)]).content
        print(f"[Celery Task] Received analysis response from LLM.")

        # --- Parse LLM Response (Logic from /analysis route) ---
        updates = {}
        try:
            json_start = analysis_llm_response.find('{')
            json_end = analysis_llm_response.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string = analysis_llm_response[json_start : json_end + 1]
                updates = json.loads(json_string)
                print(f"[Celery Task] Successfully parsed JSON updates: {updates}")
            else:
                print(f"[Celery Task] No valid JSON found in LLM analysis response.")

        except json.JSONDecodeError as e:
            print(f"[Celery Task] Failed to parse JSON from LLM response: {e}")
            print(f"[Celery Task] LLM response was: {analysis_llm_response}")

        # --- Apply Updates to User Persona State in Supabase (Logic from /analysis route) ---
        if updates:
            valid_columns = [
                'attachment_level', 'trust_level', 'mood', 'introversion_level',
                'relationship_stage', 'openness', 'energy_level', 'user_summary', 'user_facts'
            ]
            filtered_updates = {k: v for k, v in updates.items() if k in valid_columns}

            if filtered_updates:
                try:
                    print(f"[Celery Task] Attempting to update user persona state.")
                    # Fetch the current state again to apply increments/decrements correctly
                    current_state_response = supabase.table(USER_PERSONA_STATES_TABLE) \
                        .select('*') \
                        .eq('user_id', user_id_str) \
                        .eq('persona_id', ai_id) \
                        .single() \
                        .execute()

                    current_state_data = current_state_response.data if current_state_response.data else {}
                    final_updates = {}

                    for key, value in filtered_updates.items():
                        if isinstance(value, str) and (value == '+1' or value == '-1'):
                            current_value = current_state_data.get(key)
                            if isinstance(current_value, int):
                                if value == '+1':
                                    final_updates[key] = min(current_value + 1, 10)
                                elif value == '-1':
                                    final_updates[key] = max(current_value - 1, 1)
                        elif key == 'user_facts' and isinstance(value, dict):
                            current_facts = current_state_data.get('user_facts', {})
                            merged_facts = {**current_facts, **value}
                            final_updates[key] = merged_facts
                        elif key in ['attachment_level', 'trust_level', 'introversion_level', 'openness', 'energy_level'] and isinstance(value, int) and 1 <= value <= 10:
                             final_updates[key] = value
                        elif key in ['mood', 'relationship_stage', 'user_summary'] and isinstance(value, str):
                            final_updates[key] = value

                    if final_updates:
                        upsert_data = {**final_updates, 'user_id': user_id_str, 'persona_id': ai_id}
                        response = supabase.table(USER_PERSONA_STATES_TABLE) \
                            .upsert(upsert_data, on_conflict='user_id,persona_id') \
                            .execute()
                        
                        print(f"[Celery Task] User persona state updated successfully.")
                        return {"status": "success", "updated_state": response.data}
                    else:
                        print(f"[Celery Task] Filtered updates resulted in no valid updates.")
                        return {"status": "success", "message": "No valid state updates determined from analysis"}

                except Exception as e:
                    print(f"[Celery Task] Error updating user persona state in database: {e}")
                    traceback.print_exc()
                    return {"status": "error", "message": f"Failed to update persona state: {str(e)}"}
            else:
                print(f"[Celery Task] Filtered updates resulted in no valid updates.")
                return {"status": "success", "message": "No valid state updates determined from analysis"}
        else:
            print(f"[Celery Task] No updates determined from LLM analysis.")
            return {"status": "success", "message": "No state updates needed based on analysis"}


    except Exception as e:
        print(f"[Celery Task] !!! UNEXPECTED EXCEPTION CAUGHT IN ANALYSIS TASK !!! : {e}")
        traceback.print_exc()
        return {"status": "error", "message": f"An internal error occurred during analysis: {str(e)}"}




def get_ai_persona(ai_id):
    return AI_PERSONAS.get(ai_id)

# --- Helper function to verify JWT and get user ID ---
def verify_jwt(auth_header):
    print("[verify_jwt] Function called.")
    if not supabase_jwt_secret:
        print("[verify_jwt] JWT secret not set. Cannot verify token.")
        return None # Cannot verify without secret

    if not auth_header:
        print("[verify_jwt] No Authorization header provided.")
        return None

    print(f"[verify_jwt] Received header: {auth_header[:100]}...") # Log part of the header
    try:
        parts = auth_header.split()
        if parts[0].lower() != 'bearer' or len(parts) != 2:
            print(f"[verify_jwt] Invalid Authorization header format: {auth_header}")
            return None

        token = parts[1]
        print(f"[verify_jwt] Token extracted (first 50 chars): {token[:50]}...")

        print("[verify_jwt] Attempting jwt.decode...")
        # Decode and verify the JWT using the Supabase JWT secret
        # The 'algorithms' list should match the algorithm used by Supabase (usually 'HS256')
        payload = jwt.decode(token, supabase_jwt_secret, algorithms=["HS256"], options={"verify_aud": False})
        print("[verify_jwt] jwt.decode successful.")

        # The 'sub' claim contains the user's UUID
        user_id = payload.get('sub')
        if user_id:
            print(f"[verify_jwt] JWT verified. User ID: {user_id}")
            return user_id
        else:
            print("[verify_jwt] JWT missing 'sub' claim.")
            return None
    except jwt.ExpiredSignatureError:
        print("[verify_jwt] JWT has expired.")
        return None
    except jwt.InvalidTokenError:
        print("[verify_jwt] Invalid JWT token.")
        return None
    except Exception as e:
        # This catches any other unexpected errors during the process
        print(f"[verify_jwt] Unexpected error: {e}", e)
        return None

@app.get('/user/profile_stats', response_model=UserProfileStatsResponse)
async def get_user_profile_stats(authorization: Optional[str] = Header(None)):
    print(f"[GET] Entering /user/profile_stats route.")

    # 1. Verify JWT and get user ID
    user_id = verify_jwt(authorization)

    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    print(f"[GET] /user/profile_stats: User ID {user_id} authenticated.")

    if supabase is None:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        # 2. Get User Message Stats
        stats_response = supabase.table(USER_MESSAGE_STATS_TABLE) \
            .select('daily_count, total_count', 'last_reset_at') \
            .eq('user_id', user_id) \
            .single() \
            .execute()

        stats_data = stats_response.data
        daily_count = stats_data.get('daily_count', 0) if stats_data else 0
        total_count = stats_data.get('total_count', 0) if stats_data else 0
        last_reset_at = stats_data.get('last_reset_at') if stats_data else None

        # 3. Determine Plan and Get Limits
        plan_name = 'free' # Default to free
        daily_limit = 0 # Default limits
        total_limit = 0

        # Query for active subscription
        # Assuming 'status' column indicates active and current date is between start and end date
        # This is a simplified query. Real-world might need more robust date checks and status handling.
        now = datetime.utcnow().isoformat() # Use UTC time for database comparison
        subscription_response = supabase.table(USER_SUBSCRIPTIONS_TABLE) \
            .select('plan_id') \
            .eq('user_id', user_id) \
            .single() \
            .execute()

        subscription_data = subscription_response.data

        if subscription_data and 'plan_id' in subscription_data:
            # User has an active subscription, fetch plan details
            plan_id = subscription_data['plan_id']
            plan_response = supabase.table(SUBSCRIPTION_PLAN_TABLE) \
                .select('name, daily_limit, total_limit') \
                .eq('id', plan_id) \
                .single() \
                .execute()

            plan_data = plan_response.data
            if plan_data:
                plan_name = plan_data.get('name', 'Unknown Paid Plan')
                daily_limit = plan_data.get('daily_limit', 0)
                total_limit = plan_data.get('total_limit', 0)
                print(f"[GET] /user/profile_stats: User {user_id} has active plan: {plan_name}")
            else:
                 print(f"[GET] /user/profile_stats: Plan details not found for plan_id: {plan_id}")
                 # Fallback to free limits if plan details are missing
                 free_plan_response = supabase.table(SUBSCRIPTION_PLAN_TABLE) \
                     .select('daily_limit, total_limit') \
                     .eq('name', 'free') \
                     .single() \
                     .execute()
                 free_plan_data = free_plan_response.data
                 if free_plan_data:
                    daily_limit = free_plan_data.get('daily_limit', 0)
                    total_limit = free_plan_data.get('total_limit', 0)
                 plan_name = 'free (fallback)'

        else:
            # No active paid subscription, fetch free plan limits
            print(f"[GET] /user/profile_stats: User {user_id} has no active paid subscription. Fetching free plan limits.")
            free_plan_response = supabase.table(SUBSCRIPTION_PLAN_TABLE) \
                .select('daily_limit, total_limit') \
                .eq('name', 'free') \
                .single() \
                .execute()

            free_plan_data = free_plan_response.data
            if free_plan_data:
                plan_name = 'free'
                daily_limit = free_plan_data.get('daily_limit', 0)
                total_limit = free_plan_data.get('total_limit', 0)
            else:
                print(f"[GET] /user/profile_stats: 'free' plan not found in {SUBSCRIPTION_PLAN_TABLE}!")
                # Default to hardcoded limits if free plan is not defined
                plan_name = 'free'
                daily_limit = 20 # Hardcoded fallback limits
                total_limit = 500 # Hardcoded fallback limits


        # 4. Construct Response
        profile_stats = {
            'plan_type': plan_name,
            'daily_message_count': daily_count,
            'daily_message_limit': daily_limit,
            'total_message_count': total_count,
            'total_message_limit': total_limit,
            'last_reset_at': last_reset_at
        }

        print(f"[GET] /user/profile_stats: Returning stats: {profile_stats}")
        return UserProfileStatsResponse(**profile_stats)

    except HTTPException as e: # Re-raise HTTPExceptions
        raise e
    except Exception as e:
        print(f"[GET] /user/profile_stats: Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to fetch user profile data")




# --- Endpoint to list AI models (personas) ---
@app.get('/models', response_model=List[ModelInfo])
async def list_models():
    print(f"[GET] /models: Entering route.")
    """Returns a list of available AI models (personas)."""
    models_list = []
    for _id, details in AI_PERSONAS.items():
        profile_pic_url = details['profile_pic_url']


        models_list.append({
            'id': details['id'],
            'name': details['name'],
            'description': details['description'],
            'avatar_initial': details['avatar_initial'], # Still include initial as fallback
            'profile_pic_url': profile_pic_url, # Include the direct URL
        })

    print(f"[GET] /models: Returning {len(models_list)} models.")
    # Return the list as JSON
    return models_list # FastAPI handles serialization of list of dicts/Pydantic models

# --- New Endpoint to get chat history ---
@app.get('/messages/{ai_id}', response_model=List[Message])
async def get_messages_fastapi(ai_id: str, authorization: Optional[str] = Header(None)):
    print(f"[GET] /messages/{ai_id}: Entering route.")
    if supabase is None:
        raise HTTPException(status_code=503, detail="Database is not initialized.")

    user_id = verify_jwt(authorization)

    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required or invalid token.")

    print(f"[GET] /messages/{ai_id}: JWT verified for user: {user_id}. Fetching messages.")

    try:
        print(f"[GET] /messages/{ai_id}: Constructing Supabase query...")
        # Use parentheses for chained methods across lines
        response = (
            supabase.table('messages') # Using .table() consistently
            .select('id, user_id, ai_id, content, created_at, is_from_user')
            .eq('user_id', user_id)
            .eq('ai_id', ai_id)
            .order('created_at', desc=False)
            .execute() # <-- This might raise an exception on failure in your setup
        )
        print(f"[GET] /messages/{ai_id}: Supabase query execution requested. Proceeding...")

        messages_data = response.data if response.data is not None else []
        print(f"[GET] /messages/{ai_id}: Successfully processed query response. Found {len(messages_data)} messages.")

        return messages_data

    except Exception as e:
        print(f"[GET] !!! UNEXPECTED EXCEPTION IN GET_MESSAGES ROUTE (FastAPI) !!! : {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f'An internal error occurred: {str(e)}')

additional_prompt = """
You are given a history of you past conversation with the user and a message user has asked you now, 
if history is empty or irrelevant ignore it and answer user's question without using it. 
Otherwise use history as context and answer the user's question.
"""

# --- Endpoint for Chat interaction ---
@app.post('/chat', response_model=ChatResponse, responses={403: {"model": MessageLimitErrorResponse}})
async def chat(chat_request_data: ChatRequest, authorization: Optional[str] = Header(None)):
    print(f"[POST] /chat: Entering route.")

    if llm is None:
        raise HTTPException(status_code=503, detail="AI model is not initialized (API key missing or invalid).")
    if supabase is None:
        raise HTTPException(status_code=503, detail="Database is not initialized.")

    # --- SECURITY: Verify JWT and get user ID ---
    print(f"[POST] /chat: Attempting JWT verification.")
    user_id = verify_jwt(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required or invalid token.")
    print(f"[POST] /chat: JWT verified for user: {user_id}.")

    can_send, stats_or_error = check_message_limits(user_id)
    if not can_send:
        # Ensure check_message_limits returns string for error message if can_send is False
        if stats_or_error is None: # check_message_limits had an internal issue
            raise HTTPException(status_code=503, detail="Service temporarily unavailable, could not check message limits.")
        raise HTTPException(status_code=403, detail=stats_or_error)

    free_messages_today = stats_or_error['free_messages_today']
    total_messages = stats_or_error['total_messages']

    try:
        print(f"[POST] /chat: Attempting to parse JSON request body.")
        print(f"[POST] /chat: JSON body parsed successfully.")

        if not chat_request_data:
            print(f"[POST] /chat: Request body is empty.")
            raise HTTPException(status_code=400, detail="Invalid request body. Body cannot be empty.")
        

        ai_id = chat_request_data.ai_id
        user_prompt = chat_request_data.prompt
        frontend_history = chat_request_data.history

        print(f"[POST] /chat: Received prompt for AI '{ai_id}'. Prompt preview: '{user_prompt[:50]}...'")

        print(f"[POST] /chat: Received frontend history ({len(frontend_history)} messages):")
        for i, msg in enumerate(frontend_history):
             print(f"  Message {i}: sender={msg.get('sender', 'unknown')}, timestamp={msg.get('timestamp', 'N/A')}, content='{msg.get('text', '')[:50]}...'")

        persona = get_ai_persona(ai_id)
        if not persona:
            raise HTTPException(status_code=404, detail=f'AI persona with ID "{ai_id}" not found')

        user_persona_state = None
        try:
            print(f"[POST] /chat: Fetching user persona state for user {user_id} and persona {ai_id}.")
            state_response = (supabase.table(USER_PERSONA_STATES_TABLE)
                                    .select('*') 
                                    .eq('user_id', str(user_id))
                                    .eq('persona_id', ai_id)
                                    .single()
                                    .execute())

            if state_response.data:
                user_persona_state = state_response.data
                print(f"[POST] /chat: Successfully fetched user persona state: {user_persona_state}")
            else:
                print(f"[POST] /chat: No user persona state found for user {user_id} and persona {ai_id}. Creating a new one.")
                # Create a default state if none exists
                default_state = {
                    'user_id': str(user_id),
                    'persona_id': ai_id,
                    'attachment_level': 1, 
                    'trust_level': 1, 
                    'mood': 'neutral',
                    'introversion_level': 1, 
                    'relationship_stage': 'stranger',
                    'openness': 1, 
                    'energy_level': 1, 
                    'user_summary': '', 
                    'user_facts': {}
                }
                insert_response = supabase.table(USER_PERSONA_STATES_TABLE).insert([default_state]).execute()
                user_persona_state = insert_response.data[0] if insert_response.data else default_state # Use inserted data if available
                print(f"[POST] /chat: Created new user persona state: {user_persona_state}")

        except Exception as e:
            print(f"[POST] /chat: Error fetching/creating user persona state: {e}", e)
            # Continue without persona state if fetching fails, but log the error
            user_persona_state = None

        # --- 1) Embed & save the user message ---
        user_emb = get_embedding(user_prompt)
        supabase.table('messages').insert([{
            'user_id': str(user_id),
            'ai_id': ai_id,
            'content': user_prompt,
            'is_from_user': True,
            'embedding': user_emb
        }]).execute()

        # --- 2) Find top1 match and fetch 5 before/after with roles ---
        context_messages = []
        try:
            # First, find the most similar message using vector search
            print(f"[POST] /chat: Finding most relevant message...")
            most_similar_response = supabase.rpc('match_messages_by_embedding', {
                'input_user_id': str(user_id),
                'input_ai_id': ai_id,
                'query_embedding': user_emb,
                'match_count': 1
            }).execute()

            if most_similar_response.data and len(most_similar_response.data) > 0:
                most_similar_msg = most_similar_response.data[0]
                most_similar_id = most_similar_msg['id']
                most_similar_created_at = most_similar_msg['created_at']

                print(f"[POST] /chat: Found most similar message ID: {most_similar_id}")

                # Get messages before the most similar message (up to 3)
                before_response = (
                    supabase.table('messages')
                    .select('content, is_from_user, created_at')
                    .eq('user_id', str(user_id))
                    .eq('ai_id', ai_id)
                    .lt('created_at', most_similar_created_at)  # Messages before the most similar
                    .order('created_at', desc=True)
                    .limit(3)
                    .execute()
                )
                            
                # Get messages after the most similar message (up to 3)
                after_response = (
                    supabase.table('messages')
                    .select('content, is_from_user, created_at')
                    .eq('user_id', str(user_id))
                    .eq('ai_id', ai_id)
                    .gt('created_at', most_similar_created_at)  # Messages after the most similar
                    .order('created_at', desc=False)
                    .limit(3)
                    .execute()
                )
                
                # Combine all messages and sort by created_at
                before_messages = before_response.data if before_response.data else []
                similar_msg = most_similar_response.data if most_similar_response.data else []
                after_messages = after_response.data if after_response.data else []
                
                # Reverse the before messages to get chronological order
                before_messages = list(reversed(before_messages))
                
                # Combine all messages in chronological order
                context_messages = before_messages + similar_msg + after_messages

                print(f"[POST] /chat: Built context with {len(context_messages)} messages " +
                      f"({len(before_messages)} before, 1 most similar, {len(after_messages)} after)")

                print(context_messages)

        except Exception as e:
            print(f"[POST] /chat: Error fetching context messages: {e}", e)

        # --- 3) Assemble history ---
        history = []
        for msg in context_messages:
            role = 'user' if msg['is_from_user'] else ai_id
            history.append({'role': role, 'content': msg['content']})

        # --- 5) Build messages for LLM ---
        chat_msgs = [SystemMessage(content=additional_prompt)]
        chat_msgs.append(SystemMessage(content=persona['instruction']))

        if user_persona_state:
            state_context = f"""
            User's current state with this persona (ID: {user_persona_state.get('persona_id', 'N/A')}):
            - Attachment Level: {user_persona_state.get('attachment_level', 'N/A')} (1-10)
            - Trust Level: {user_persona_state.get('trust_level', 'N/A')} (1-10)
            - Mood: {user_persona_state.get('mood', 'N/A')}
            - Introversion Level: {user_persona_state.get('introversion_level', 'N/A')} (1-10)
            - Relationship Stage: {user_persona_state.get('relationship_stage', 'N/A')}
            - Openness: {user_persona_state.get('openness', 'N/A')} (1-10)
            - Energy Level: {user_persona_state.get('energy_level', 'N/A')} (1-10)
            - User Summary: {user_persona_state.get('user_summary', 'N/A')}
            - User Facts: {json.dumps(user_persona_state.get('user_facts', {}))}
            Use this information to inform your response and maintain context about your relationship with the user.
            """
            chat_msgs.append(SystemMessage(content=state_context))
            print(f"[POST] /chat: Added user persona state to LLM context.")

        if history:
            chat_msgs.append(SystemMessage(content="Long term context:"))
            for msg in history:
                if msg['role'] == 'user':
                    chat_msgs.append(HumanMessage(content=msg['content']))
                else:
                    chat_msgs.append(AIMessage(content=msg['content'], role=msg['role']))
        
        frontend_history_for_llm = []
        for msg in frontend_history:
            # Basic validation for required fields in history messages from frontend
            if 'text' in msg and 'sender' in msg:
                # Map frontend 'sender' to backend 'role'
                role = 'user' if msg['sender'] == 'user' else ai_id # Assuming frontend 'ai' sender maps to backend ai_id role
                frontend_history_for_llm.append({'role': role, 'content': msg['text']})
            else:
                print(f"[POST] /chat: Skipping malformed frontend history message: {msg}")

        print("Frontend History for LLM:", frontend_history_for_llm) 

        if frontend_history_for_llm:
            chat_msgs.append(SystemMessage(content="Most Recent Conversation History:"))
            for msg in frontend_history_for_llm:
                 # Use HumanMessage for 'user' role, AIMessage for AI role (using ai_id as role name)
                 if msg['role'] == 'user':
                     chat_msgs.append(HumanMessage(content=msg['content']))
                 else: # Assuming any other role is the AI role (ai_id)
                     chat_msgs.append(AIMessage(content=msg['content'], role=msg['role'])) # Pass role to AIMessage

        chat_msgs.append(SystemMessage(content="User Prompt:"))
        chat_msgs.append(HumanMessage(content=user_prompt))

        print(f"[POST] /chat: Sending {chat_msgs} messages to LLM.")

        # llm_resp = llm.invoke(chat_msgs).content

        try:
            # Potentially enable Langchain debug mode for more verbose output globally
            # import langchain
            # langchain.debug = True # Or set LANGCHAIN_DEBUG=true environment variable

            llm_response_object = llm.invoke(chat_msgs)
            llm_resp = llm_response_object.content
            print(f"[POST] /chat: Successfully received LLM response object: {llm_response_object}")

        except Exception as e:
            print(f"[POST] !!! EXCEPTION DURING llm.invoke() !!!")
            print(f"[POST] Exception Type: {type(e)}")
            print(f"[POST] Exception Args: {e.args}")
            print(f"[POST] Exception Details: {e}") # Full exception string

        print("got llm response:", llm_resp)
        # --- 6) Save AI response + its embedding ---
        ai_emb = get_embedding(llm_resp)
        supabase.table('messages').insert([{
            'user_id': str(user_id),
            'ai_id': ai_id,
            'content': llm_resp,
            'is_from_user': False,
            'embedding': ai_emb
        }]).execute()
        print(f"[POST] /chat: AI response saved successfully.")

        # --- 7) Update message stats ---
        if free_messages_today < 20:
            supabase.table('user_message_stats') \
              .update({'daily_count': free_messages_today + 1}) \
              .eq('user_id', user_id).execute()
        supabase.table('user_message_stats') \
          .update({'total_count': total_messages + 1}) \
          .eq('user_id', user_id).execute()
        print(f"[POST] /chat: Message stats updated successfully.")
        # --- 8) Perform analysis task ---
        print(f"[POST] /chat: Starting analysis task...")
        perform_analysis_task.delay(str(user_id), ai_id, user_prompt)
        
        return ChatResponse(response=llm_resp)

    except HTTPException as e: # Re-raise HTTPExceptions
        raise e
    except Exception as e:
        print(f"[POST] !!! UNEXPECTED EXCEPTION CAUGHT IN CHAT ROUTE (FastAPI) !!! : {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f'An internal error occurred: {str(e)}')


if __name__ == '__main__':
    # Ensure host is '0.0.0.0' to be accessible from emulator/other devices
    print("Starting FastAPI development server...")
    uvicorn.run(app, host='0.0.0.0', port=5000, log_level="debug")
