# app/api/routes_messages.py
from fastapi import APIRouter, HTTPException, Header
from typing import List, Optional
import sys
import traceback

from app.models.pydantic_models import Message
from app.core.security import verify_jwt_token
from app.services.supabase_client import supabase # Shared Supabase client
from app.core.config import MESSAGES_TABLE

router = APIRouter()

@router.get("/messages/{ai_id}", response_model=List[Message])
async def get_chat_messages_route(ai_id: str, authorization: Optional[str] = Header(None)):
    print(f"[API] GET /messages/{ai_id} called.")
    sys.stdout.flush()

    if not supabase:
        raise HTTPException(status_code=503, detail="Database service not available.")

    user_id = verify_jwt_token(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing token")

    print(f"[API] /messages/{ai_id}: User {user_id} authenticated.")
    sys.stdout.flush()
    
    try:
        response = supabase.table(MESSAGES_TABLE) \
            .select('id, user_id, ai_id, content, created_at, is_from_user') \
            .eq('user_id', user_id) \
            .eq('ai_id', ai_id) \
            .order('created_at', desc=False) \
            .execute()

        if response.data is not None:
            messages_data = response.data
            print(f"[API] /messages/{ai_id}: Found {len(messages_data)} messages.")
            sys.stdout.flush()
            return messages_data
        else:
            # Handle case where response.data is None (e.g., error in query, though PostgREST usually gives error object)
            print(f"[API] /messages/{ai_id}: No data returned from query, or error. Response: {response}")
            sys.stdout.flush()
            # Check for PostgREST error if needed: if getattr(response, 'error', None): raise HTTPException(...)
            return [] # Return empty list if no messages or non-critical error

    except Exception as e:
        print(f"[API] /messages/{ai_id}: Error fetching messages for user {user_id}: {e}")
        sys.stdout.flush()
        traceback.print_exc()
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=f"An internal error occurred while fetching messages.")