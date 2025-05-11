import os
import jwt
import traceback
import sys # Import sys for flushing

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

app = Flask(__name__)

# Get the Google API Key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")


supabase_url = os.getenv("SUPABASE_URL") # Get Supabase URL
supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY") # Get Supabase Service Role Key
supabase_jwt_secret = os.getenv("SUPABASE_JWT_SECRET")

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


# --- Define AI Personas (Different instructions for the same underlying model) ---
AI_PERSONAS = {
    'luffy': {
        'id': 'luffy',
        'name': 'Luffy',
        'instruction': 'You are a friendly and helpful AI assistant. Answer questions clearly and concisely.',
        'description': 'Orewa Monkey D. Luffy, the captain of the Straw Hat Pirates. I am on a quest to find the One Piece and become the Pirate King!',
        'profile_pic_filename': 'https://i.ibb.co/7tkMdVNp/luffy.jpg',
        'avatar_initial': 'L',
    },
    'naruto': {
        'id': 'naruto',
        'name': 'Naruto',
        'instruction': 'You are a creative writing assistant. Help the user brainstorm ideas, write stories, or compose poems. Be imaginative.',
        'description': 'I am naruto uzumaki, a ninja from the Hidden Leaf Village. I dream of becoming Hokage and protecting my friends!',
        'profile_pic_filename': 'https://i.ibb.co/spYVTRzx/naruto.jpg',
        'avatar_initial': 'H',
    },
    'zoro': {
        'id': 'zoro',
        'name': 'Zoro',
        'instruction': 'You are a knowledgeable technical expert. Provide detailed explanations and solutions for programming and technology-related questions.',
        'description': 'I am zoro, the swordsman of the Straw Hat Pirates. I am on a quest to become the world\'s greatest swordsman!',
        'profile_pic_filename': 'https://i.ibb.co/7tR2qpjf/zoro.jpg',
        'avatar_initial': 'Z',
    },
}

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


# --- Endpoint to list AI models (personas) ---
@app.route('/models', methods=['GET'])
def list_models():
    log_info(f"[{request.method}] Entering /models route.")
    """Returns a list of available AI models (personas)."""
    models_list = []
    for ai_id, details in AI_PERSONAS.items():
        profile_pic_url = details['profile_pic_filename']
        # Static file serving was commented out, using direct URLs for now.
        # if details.get('profile_pic_filename') and base_url:
        #     # Generate the URL for the static file
        #     # url_for('static', filename=...) creates the path like /static/image.png
        #     # We prepend the base URL to make it absolute for the mobile app
        #     # Assuming base_url is configured elsewhere or handled by frontend/cdn
        #     # For now, using direct external URLs as in the personas dict
        #     pass


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
# --- Endpoint for Chat interaction ---
@app.route('/chat', methods=['POST'])
def chat():
    log_info(f"[{request.method}] Entering /chat route.")
    """
    Receives user input, saves user message, gets AI response (potentially with RAG context),
    saves AI response, and returns AI response text.
    Requires user's JWT in the Authorization header.
    """
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