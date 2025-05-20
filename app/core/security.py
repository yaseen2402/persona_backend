# app/core/security.py
import jwt
import sys
from app.core.config import SUPABASE_JWT_SECRET
from typing import Optional

def verify_jwt_token(auth_header: Optional[str]) -> Optional[str]:
    # print("[verify_jwt] Function called.") # Can be verbose
    # sys.stdout.flush()

    if not SUPABASE_JWT_SECRET:
        print("[verify_jwt] JWT secret (SUPABASE_JWT_SECRET) not set in config. Cannot verify token.")
        sys.stdout.flush()
        return None

    if not auth_header:
        # print("[verify_jwt] No Authorization header provided.") # Common, maybe log at DEBUG level
        # sys.stdout.flush()
        return None

    # print(f"[verify_jwt] Received header: {auth_header[:100]}...") # Verbose
    # sys.stdout.flush()
    try:
        parts = auth_header.split()
        if len(parts) == 1 and parts[0].lower().startswith('bearer'): # Handle cases where frontend might send "BearerTOKEN"
            token = parts[0][len('bearer'):].lstrip()
        elif parts[0].lower() != 'bearer' or len(parts) != 2:
            print(f"[verify_jwt] Invalid Authorization header format.")
            sys.stdout.flush()
            return None
        else:
            token = parts[1]
        
        # print(f"[verify_jwt] Token extracted (first 50 chars): {token[:50]}...") # Verbose
        # sys.stdout.flush()

        # print("[verify_jwt] Attempting jwt.decode...") # Verbose
        # sys.stdout.flush()
        payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})
        # print("[verify_jwt] jwt.decode successful.") # Verbose
        # sys.stdout.flush()

        user_id = payload.get('sub')
        if user_id:
            # print(f"[verify_jwt] JWT verified. User ID: {user_id}") # Verbose
            # sys.stdout.flush()
            return user_id
        else:
            print("[verify_jwt] JWT missing 'sub' claim.")
            sys.stdout.flush()
            return None
    except jwt.ExpiredSignatureError:
        print("[verify_jwt] JWT has expired.")
        sys.stdout.flush()
        return None
    except jwt.InvalidTokenError as e:
        print(f"[verify_jwt] Invalid JWT token: {e}")
        sys.stdout.flush()
        return None
    except Exception as e:
        print(f"[verify_jwt] Unexpected error during JWT verification: {e}")
        sys.stdout.flush()
        # traceback.print_exc() # Consider if too verbose for common errors
        return None