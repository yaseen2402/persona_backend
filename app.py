import os
import jwt
import traceback
import sys # Import sys for flushing

import json
from pathlib import Path
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, url_for
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()


embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

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


# Custom 500 error handler - Improved logging
@app.errorhandler(500)
def handle_500_error(error):
    print(f"Caught 500 Internal Server Error: {error}", error)
    # If you comment out this handler, Flask's built-in interactive debugger might show in browser during dev.
    # Return a generic error to the client
    return jsonify({"error": "Internal Server Error"}), 500


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

@app.route('/user/profile_stats', methods=['GET'])
def get_user_profile_stats():
    print(f"[{request.method}] Entering /user/profile_stats route.")

    # 1. Verify JWT and get user ID
    auth_header = request.headers.get('Authorization')
    user_id = verify_jwt(auth_header)

    if not user_id:
        print(f"[{request.method}] /user/profile_stats: Unauthorized access - Invalid or missing token.")
        return jsonify({"error": "Unauthorized"}), 401

    print(f"[{request.method}] /user/profile_stats: User ID {user_id} authenticated.")

    if supabase is None:
        print(f"[{request.method}] /user/profile_stats: Supabase client is not initialized.")
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
                print(f"[{request.method}] /user/profile_stats: User {user_id} has active plan: {plan_name}")
            else:
                 print(f"[{request.method}] /user/profile_stats: Plan details not found for plan_id: {plan_id}")
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
            print(f"[{request.method}] /user/profile_stats: User {user_id} has no active paid subscription. Fetching free plan limits.")
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
                print(f"[{request.method}] /user/profile_stats: 'free' plan not found in {SUBSCRIPTION_PLAN_TABLE}!")
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

        print(f"[{request.method}] /user/profile_stats: Returning stats: {profile_stats}")
        return jsonify(profile_stats), 200

    except Exception as e:
        print(f"[{request.method}] /user/profile_stats: Unexpected error: {e}", e)
        # Return a generic error to the client
        return jsonify({"error": "Failed to fetch user profile data"}), 500




# --- Endpoint to list AI models (personas) ---
@app.route('/models', methods=['GET'])
def list_models():
    print(f"[{request.method}] Entering /models route.")
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

    print(f"[{request.method}] Returning {len(models_list)} models.")
    # Return the list as JSON
    return jsonify(models_list)

# --- New Endpoint to get chat history ---
@app.route('/messages/<ai_id>', methods=['GET'])
def get_messages(ai_id):
    print(f"[{request.method}] Entering /messages/{ai_id} route.")
    if supabase is None:
        print(f"[{request.method}] Supabase is not initialized.")
        return jsonify({'error': 'Database is not initialized.'}), 503

    print(f"[{request.method}] Attempting JWT verification.")
    auth_header = request.headers.get('Authorization')
    user_id = verify_jwt(auth_header)

    if not user_id:
        print(f"[{request.method}] Authentication failed or no token.")
        return jsonify({'error': 'Authentication required or invalid token.'}), 401

    print(f"[{request.method}] JWT verified for user: {user_id}. Fetching messages.")

    try:
        print(f"[{request.method}] Constructing Supabase query...")
        # Use parentheses for chained methods across lines
        response = (
            supabase.table('messages') # Using .table() consistently
            .select('id, user_id, ai_id, content, created_at, is_from_user')
            .eq('user_id', user_id)
            .eq('ai_id', ai_id)
            .order('created_at', desc=False)
            .execute() # <-- This might raise an exception on failure in your setup
        )
        print(f"[{request.method}] Supabase query execution requested. Proceeding...")

        
        messages_data = response.data if response.data is not None else []
        print(f"[{request.method}] Successfully processed query response. Found {len(messages_data)} messages.")

        return jsonify(messages_data)

    except Exception as e:
        print(f"[{request.method}] !!! UNEXPECTED EXCEPTION CAUGHT IN GET_MESSAGES ROUTE !!! : {e}", e)
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

additional_prompt = """
You are given a history of you past conversation with the user and a message user has asked you now, 
if history is empty or irrelevant ignore it and answer user's question without using it. 
Otherwise use history as context and answer the user's question.
"""

def format_vector_for_query(vector_list):
    """Formats a Python list as a PostgreSQL vector literal string"""
    # Convert the list to a proper PostgreSQL vector syntax '[x1,x2,x3,...]'
    vector_str = '[' + ','.join(str(x) for x in vector_list) + ']'
    return vector_str

# --- Endpoint for Chat interaction ---
@app.route('/chat', methods=['POST'])
def chat():
    print(f"[{request.method}] Entering /chat route.")
    
    if llm is None:
        print(f"[{request.method}] LLM is not initialized.")
        return jsonify({'error': 'AI model is not initialized (API key missing or invalid).'}), 503 # Service Unavailable
    if supabase is None:
        print(f"[{request.method}] Supabase is not initialized.")
        return jsonify({'error': 'Database is not initialized.'}), 503

    # --- SECURITY: Verify JWT and get user ID ---
    print(f"[{request.method}] Attempting JWT verification.")
    auth_header = request.headers.get('Authorization')
    user_id = verify_jwt(auth_header)
    if not user_id:
        print(f"[{request.method}] Authentication failed or no token.")
        return jsonify({'error': 'Authentication required or invalid token.'}), 401
    print(f"[{request.method}] JWT verified for user: {user_id}.")
    
    can_send, stats_or_error = check_message_limits(user_id)
    if not can_send:
        return jsonify({'error': stats_or_error}), 403  # Bad Request: Message limit exceeded
    
    free_messages_today = stats_or_error['free_messages_today']
    total_messages = stats_or_error['total_messages']

    try:
        print(f"[{request.method}] Attempting to parse JSON request body.")
        data = request.get_json()
        print(f"[{request.method}] JSON body parsed successfully.")

        if not data:
            print(f"[{request.method}] Request body is empty.")
            return jsonify({'error': 'Invalid request body. Body cannot be empty.'}), 400
        if 'ai_id' not in data or 'prompt' not in data:
            print(f"[{request.method}] Missing 'ai_id' or 'prompt' in request body. Data: {data}")
            return jsonify({'error': 'Invalid request body. "ai_id" and "prompt" fields are required.'}), 400

        ai_id = data['ai_id']
        user_prompt = data['prompt']
        frontend_history = data.get('history', [])

        print(f"[{request.method}] Received prompt for AI '{ai_id}'. Prompt preview: '{user_prompt[:50]}...'")

        print(f"[{request.method}] Received frontend history ({len(frontend_history)} messages):")
        for i, msg in enumerate(frontend_history):
             print(f"  Message {i}: sender={msg.get('sender', 'unknown')}, timestamp={msg.get('timestamp', 'N/A')}, content='{msg.get('text', '')[:50]}...'")

        persona = get_ai_persona(ai_id)
        if not persona:
            print(f"[{request.method}] AI persona with ID '{ai_id}' not found.")
            return jsonify({'error': f'AI persona with ID "{ai_id}" not found'}), 404

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
            print(f"[{request.method}] Finding most relevant message...")
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
                
                print(f"[{request.method}] Found most similar message ID: {most_similar_id}")
                
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
                
                print(f"[{request.method}] Built context with {len(context_messages)} messages " +
                      f"({len(before_messages)} before, 1 most similar, {len(after_messages)} after)")

                print(context_messages)

        except Exception as e:
            print(f"[{request.method}] Error fetching context messages: {e}", e)

        # --- 3) Assemble history ---
        history = []
        for msg in context_messages:
            role = 'user' if msg['is_from_user'] else ai_id
            history.append({'role': role, 'content': msg['content']})

        # --- 5) Build messages for LLM ---
        chat_msgs = [SystemMessage(content=additional_prompt)]
        chat_msgs.append(SystemMessage(content=persona['instruction']))

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
                print(f"[{request.method}] Skipping malformed frontend history message: {msg}")

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
        
        print(f"[{request.method}] Sending {chat_msgs} messages to LLM.")

        llm_resp = llm.invoke(chat_msgs).content

        # --- 6) Save AI response + its embedding ---
        ai_emb = get_embedding(llm_resp)
        supabase.table('messages').insert([{
            'user_id': str(user_id),
            'ai_id': ai_id,
            'content': llm_resp,
            'is_from_user': False,
            'embedding': ai_emb
        }]).execute()

        # --- 7) Update message stats ---
        if free_messages_today < 20:
            supabase.table('user_message_stats') \
              .update({'daily_count': free_messages_today + 1}) \
              .eq('user_id', user_id).execute()
        supabase.table('user_message_stats') \
          .update({'total_count': total_messages + 1}) \
          .eq('user_id', user_id).execute()

        return jsonify({'response': llm_resp})

    except Exception as e:
        # This block will now catch *any* exception during the try block, including
        # potential exceptions raised directly by the .execute() calls if they fail.
        print(f"[{request.method}] !!! UNEXPECTED EXCEPTION CAUGHT IN CHAT ROUTE !!! : {e}", e)
        return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500
    
if __name__ == '__main__':
    # Run the Flask development server
    # Ensure host is '0.0.0.0' to be accessible from emulator/other devices
    print("Starting Flask development server...")
    app.run(host='0.0.0.0', debug=True, port=5000)
    print("Flask server stopped.")