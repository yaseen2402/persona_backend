import os
import jwt
import traceback
import sys # Import sys for flushing

import json
from pathlib import Path
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, url_for
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage # Import message types
from dotenv import load_dotenv
from supabase import create_client, Client
# Load environment variables from .env file
load_dotenv()

# --- Logging Helper ---
def log_info(message):
    """Helper for standard info logging."""
    print(f"[INFO] {message}", flush=True)

def log_warning(message):
    """Helper for warning logging."""
    print(f"[WARNING] {message}", file=sys.stderr, flush=True)

def log_error(message, exception=None):
    """Helper for error logging, includes traceback if exception is provided."""
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)
    if exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()

def check_message_limits(user_id):
    log_info(f"[check_message_limits] Checking message limits for user {user_id}.")
    
    if supabase is None:
        log_error("[check_message_limits] Supabase is not initialized.")
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
            log_warning(f"[check_message_limits] No data found for user {user_id}.")
            return None, None

        stats = response.data
        free_messages_today = stats['daily_count']
        total_messages = stats['total_count']
        
        # Check if user has remaining free messages for today
        if free_messages_today >= 20:
            log_warning(f"[check_message_limits] User {user_id} has exceeded the daily free messages limit.")
            return False, "Daily free message limit exceeded."
        
        # Check if total message count exceeds 500
        if total_messages >= 500:
            log_warning(f"[check_message_limits] User {user_id} has exceeded the total message limit.")
            return False, "Total message limit exceeded."

        return True, {'free_messages_today': free_messages_today, 'total_messages': total_messages}
        
    except Exception as e:
        log_error(f"[check_message_limits] Error checking message limits for user {user_id}: {e}", e)
        return False, "An error occurred while checking your message limits."
    
app = Flask(__name__)

# Get the Google API Key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")


supabase_url = os.getenv("SUPABASE_URL") # Get Supabase URL
supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY") # Get Supabase Service Role Key
supabase_jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
PERSONAS_PATH = Path("personas.json")

SUBSCRIPTION_PLAN_TABLE = 'subscription_plans'
USER_MESSAGE_STATS_TABLE = 'user_message_stats'
USER_SUBSCRIPTIONS_TABLE = 'user_subscriptions'

with open(PERSONAS_PATH, "r", encoding="utf-8") as f:
    persona_list = json.load(f)
    AI_PERSONAS = {p["id"]: p for p in persona_list}


# --- Initialize Supabase Client (Backend) ---
supabase = None
if not supabase_url or not supabase_service_key:
    log_error("SUPABASE_URL or SUPABASE_SERVICE_KEY environment variable not set.")
    log_warning("Database interaction features will be disabled.")
else:
    try:
        # Use the service role key for backend operations (bypasses RLS)
        supabase = create_client(supabase_url, supabase_service_key)
        log_info("Supabase client initialized successfully on the backend.")
    except Exception as e:
        supabase = None
        log_error(f"Error initializing Supabase client on backend: {e}", e)
        log_warning("Database interaction features will be disabled.")

llm = None  # type: ChatGoogleGenerativeAI | None

if not google_api_key:
    log_error("GOOGLE_API_KEY environment variable not set.")
    log_warning("AI chat functionality will be disabled.")
else:
    # Initialize the Langchain ChatGoogleGenerativeAI model
    # Specify the model name
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key)
        log_info("Gemini model initialized successfully.")
        # Optional: Test a simple invoke to check if key/model works
        # try:
        #     test_response = llm.invoke("Hello, test.")
        #     log_info(f"Gemini test invoke successful. Response preview: {test_response.content[:50]}...")
        # except Exception as test_e:
        #      log_warning(f"Gemini test invoke failed: {test_e}. API key or model might have issues despite initialization.")

    except Exception as e:
        llm = None
        log_error(f"Error initializing Gemini model: {e}", e)
        log_warning("AI chat functionality will be disabled.")


# Custom 500 error handler - Improved logging
@app.errorhandler(500)
def handle_500_error(error):
    log_error(f"Caught 500 Internal Server Error: {error}", error)
    # If you comment out this handler, Flask's built-in interactive debugger might show in browser during dev.
    # Return a generic error to the client
    return jsonify({"error": "Internal Server Error"}), 500


def get_ai_persona(ai_id):
    return AI_PERSONAS.get(ai_id)

# --- Helper function to verify JWT and get user ID ---
def verify_jwt(auth_header):
    log_info("[verify_jwt] Function called.")
    if not supabase_jwt_secret:
        log_error("[verify_jwt] JWT secret not set. Cannot verify token.")
        return None # Cannot verify without secret

    if not auth_header:
        log_warning("[verify_jwt] No Authorization header provided.")
        return None

    log_info(f"[verify_jwt] Received header: {auth_header[:100]}...") # Log part of the header
    try:
        parts = auth_header.split()
        if parts[0].lower() != 'bearer' or len(parts) != 2:
            log_warning(f"[verify_jwt] Invalid Authorization header format: {auth_header}")
            return None

        token = parts[1]
        log_info(f"[verify_jwt] Token extracted (first 50 chars): {token[:50]}...")

        log_info("[verify_jwt] Attempting jwt.decode...")
        # Decode and verify the JWT using the Supabase JWT secret
        # The 'algorithms' list should match the algorithm used by Supabase (usually 'HS256')
        payload = jwt.decode(token, supabase_jwt_secret, algorithms=["HS256"], options={"verify_aud": False})
        log_info("[verify_jwt] jwt.decode successful.")

        # The 'sub' claim contains the user's UUID
        user_id = payload.get('sub')
        if user_id:
            log_info(f"[verify_jwt] JWT verified. User ID: {user_id}")
            return user_id
        else:
            log_warning("[verify_jwt] JWT missing 'sub' claim.")
            return None
    except jwt.ExpiredSignatureError:
        log_warning("[verify_jwt] JWT has expired.")
        return None
    except jwt.InvalidTokenError:
        log_warning("[verify_jwt] Invalid JWT token.")
        return None
    except Exception as e:
        # This catches any other unexpected errors during the process
        log_error(f"[verify_jwt] Unexpected error: {e}", e)
        return None

@app.route('/user/profile_stats', methods=['GET'])
def get_user_profile_stats():
    log_info(f"[{request.method}] Entering /user/profile_stats route.")

    # 1. Verify JWT and get user ID
    auth_header = request.headers.get('Authorization')
    user_id = verify_jwt(auth_header)

    if not user_id:
        log_warning(f"[{request.method}] /user/profile_stats: Unauthorized access - Invalid or missing token.")
        return jsonify({"error": "Unauthorized"}), 401

    log_info(f"[{request.method}] /user/profile_stats: User ID {user_id} authenticated.")

    if supabase is None:
        log_error(f"[{request.method}] /user/profile_stats: Supabase client is not initialized.")
        return jsonify({"error": "Database not available"}), 500

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
                log_info(f"[{request.method}] /user/profile_stats: User {user_id} has active plan: {plan_name}")
            else:
                 log_warning(f"[{request.method}] /user/profile_stats: Plan details not found for plan_id: {plan_id}")
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
            log_info(f"[{request.method}] /user/profile_stats: User {user_id} has no active paid subscription. Fetching free plan limits.")
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
                log_error(f"[{request.method}] /user/profile_stats: 'free' plan not found in {SUBSCRIPTION_PLAN_TABLE}!")
                # Default to hardcoded limits if free plan is not defined
                plan_name = 'free (limits not defined)'
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

        log_info(f"[{request.method}] /user/profile_stats: Returning stats: {profile_stats}")
        return jsonify(profile_stats), 200

    except Exception as e:
        log_error(f"[{request.method}] /user/profile_stats: Unexpected error: {e}", e)
        # Return a generic error to the client
        return jsonify({"error": "Failed to fetch user profile data"}), 500




# --- Endpoint to list AI models (personas) ---
@app.route('/models', methods=['GET'])
def list_models():
    log_info(f"[{request.method}] Entering /models route.")
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

    log_info(f"[{request.method}] Returning {len(models_list)} models.")
    # Return the list as JSON
    return jsonify(models_list)

# --- New Endpoint to get chat history ---
@app.route('/messages/<ai_id>', methods=['GET'])
def get_messages(ai_id):
    log_info(f"[{request.method}] Entering /messages/{ai_id} route.")
    if supabase is None:
        log_error(f"[{request.method}] Supabase is not initialized.")
        return jsonify({'error': 'Database is not initialized.'}), 503

    log_info(f"[{request.method}] Attempting JWT verification.")
    auth_header = request.headers.get('Authorization')
    user_id = verify_jwt(auth_header)

    if not user_id:
        log_warning(f"[{request.method}] Authentication failed or no token.")
        return jsonify({'error': 'Authentication required or invalid token.'}), 401

    log_info(f"[{request.method}] JWT verified for user: {user_id}. Fetching messages.")

    try:
        log_info(f"[{request.method}] Constructing Supabase query...")
        # Use parentheses for chained methods across lines
        response = (
            supabase.table('messages') # Using .table() consistently
            .select('id, user_id, ai_id, content, created_at, is_from_user')
            .eq('user_id', user_id)
            .eq('ai_id', ai_id)
            .order('created_at', desc=False)
            .execute() # <-- This might raise an exception on failure in your setup
        )
        log_info(f"[{request.method}] Supabase query execution requested. Proceeding...")

        # --- Removed explicit status check: if response.status_code != 200: ---
        # Relying on the outer try...except to catch errors from .execute()


        # If no messages are found, response.data is None or empty list depending on supabase-py version
        # Accessing .data should be safe if execute() didn't raise an exception
        messages_data = response.data if response.data is not None else []
        log_info(f"[{request.method}] Successfully processed query response. Found {len(messages_data)} messages.")

        return jsonify(messages_data)

    except Exception as e:
        # This block will now catch any exception during the try block, including
        # potential exceptions raised directly by the .execute() call if it fails.
        log_error(f"[{request.method}] !!! UNEXPECTED EXCEPTION CAUGHT IN GET_MESSAGES ROUTE !!! : {e}", e)
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

# --- Endpoint for Chat interaction ---
@app.route('/chat', methods=['POST'])
def chat():
    log_info(f"[{request.method}] Entering /chat route.")
    
    if llm is None:
        log_error(f"[{request.method}] LLM is not initialized.")
        return jsonify({'error': 'AI model is not initialized (API key missing or invalid).'}), 503 # Service Unavailable
    if supabase is None:
        log_error(f"[{request.method}] Supabase is not initialized.")
        return jsonify({'error': 'Database is not initialized.'}), 503

    # --- SECURITY: Verify JWT and get user ID ---
    log_info(f"[{request.method}] Attempting JWT verification.")
    auth_header = request.headers.get('Authorization')
    user_id = verify_jwt(auth_header)
    if not user_id:
        log_warning(f"[{request.method}] Authentication failed or no token.")
        return jsonify({'error': 'Authentication required or invalid token.'}), 401
    log_info(f"[{request.method}] JWT verified for user: {user_id}.")
    
    can_send, stats_or_error = check_message_limits(user_id)
    if not can_send:
        return jsonify({'error': stats_or_error}), 403  # Bad Request: Message limit exceeded
    
    free_messages_today = stats_or_error['free_messages_today']
    total_messages = stats_or_error['total_messages']

    try:
        log_info(f"[{request.method}] Attempting to parse JSON request body.")
        data = request.get_json()
        log_info(f"[{request.method}] JSON body parsed successfully.")

        if not data:
            log_warning(f"[{request.method}] Request body is empty.")
            return jsonify({'error': 'Invalid request body. Body cannot be empty.'}), 400
        if 'ai_id' not in data or 'prompt' not in data:
            log_warning(f"[{request.method}] Missing 'ai_id' or 'prompt' in request body. Data: {data}")
            return jsonify({'error': 'Invalid request body. "ai_id" and "prompt" fields are required.'}), 400

        ai_id = data['ai_id']
        user_prompt = data['prompt']

        log_info(f"[{request.method}] Received prompt for AI '{ai_id}'. Prompt preview: '{user_prompt[:50]}...'")

        persona = get_ai_persona(ai_id)
        if not persona:
            log_warning(f"[{request.method}] AI persona with ID '{ai_id}' not found.")
            return jsonify({'error': f'AI persona with ID "{ai_id}" not found'}), 404

        # --- Save User Message to Supabase ---
        log_info(f"[{request.method}] Attempting to save user message to Supabase for user {user_id}, AI {ai_id}.")
        user_message_data = {
            'user_id': str(user_id), # Ensure user_id is treated as string/uuid type
            'ai_id': ai_id,
            'content': user_prompt,
            'is_from_user': True,
        }
        # Use parentheses for chained methods across lines
        insert_response_user = (
            supabase.table('messages')
            .insert([user_message_data])
            .execute()
        )
        # --- Removed explicit status check: if insert_response_user.status_code != 201: ---
        log_info(f"[{request.method}] User message insert execution requested. Proceeding...")


        # --- Get AI Response ---
        log_info(f"[{request.method}] Preparing LLM input and attempting invoke.")
        system_instruction = persona['instruction']
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_prompt)
        ]
        response_from_llm = llm.invoke(messages)
        gemini_response = response_from_llm.content
        log_info(f"[{request.method}] LLM invoke successful. Response content preview: {gemini_response[:50]}...")

        # --- Save AI Response to Supabase ---
        log_info(f"[{request.method}] Attempting to save AI response to Supabase for user {user_id}, AI {ai_id}.")
        ai_message_data = {
            'user_id': str(user_id), # Ensure user_id is treated as string/uuid type
            'ai_id': ai_id,
            'content': gemini_response,
            'is_from_user': False,
        }
        # Use parentheses for chained methods across lines
        insert_response_ai = (
            supabase.table('messages')
            .insert([ai_message_data])
            .execute()
        )
        # --- Removed explicit status check: if insert_response_ai.status_code != 201: ---
        log_info(f"[{request.method}] AI message insert execution requested. Proceeding...")

        if free_messages_today < 20:
            new_free_messages = free_messages_today + 1
            supabase.table('user_message_stats') \
                .update({'daily_count': new_free_messages}) \
                .eq('user_id', user_id) \
                .execute()

        # Update the total messages count
        supabase.table('user_message_stats') \
            .update({'total_count': total_messages + 1}) \
            .eq('user_id', user_id) \
            .execute()
        
        log_info(f"[{request.method}] Message processing complete. Returning response.")
        return jsonify({'response': gemini_response})

    except Exception as e:
        # This block will now catch *any* exception during the try block, including
        # potential exceptions raised directly by the .execute() calls if they fail.
        log_error(f"[{request.method}] !!! UNEXPECTED EXCEPTION CAUGHT IN CHAT ROUTE !!! : {e}", e)
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500
    
if __name__ == '__main__':
    # Run the Flask development server
    # Ensure host is '0.0.0.0' to be accessible from emulator/other devices
    log_info("Starting Flask development server...")
    app.run(host='0.0.0.0', debug=True, port=5000)
    log_info("Flask server stopped.")