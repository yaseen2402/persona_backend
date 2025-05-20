# app/api/routes_users.py
from fastapi import APIRouter, HTTPException, Header
from typing import Optional
import sys
import traceback

from app.models.pydantic_models import UserProfileStatsResponse
from app.core.security import verify_jwt_token
from app.services.user_service import get_user_profile_stats_logic # Business logic

router = APIRouter()

@router.get("/user/profile_stats", response_model=UserProfileStatsResponse)
async def get_user_profile_stats_route(authorization: Optional[str] = Header(None)):
    print(f"[API] GET /user/profile_stats called.")
    sys.stdout.flush()

    user_id = verify_jwt_token(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing token")

    print(f"[API] /user/profile_stats: User {user_id} authenticated.")
    sys.stdout.flush()

    try:
        profile_stats_data = get_user_profile_stats_logic(user_id)
        if profile_stats_data is None:
            # This case means a critical error happened in the service layer (e.g., DB totally down)
            raise HTTPException(status_code=503, detail="Service temporarily unavailable while fetching profile stats.")
        
        # Convert to Pydantic model for response
        return UserProfileStatsResponse(**profile_stats_data)

    except HTTPException as e: # Re-raise HTTPExceptions explicitly
        raise e
    except Exception as e:
        print(f"[API] /user/profile_stats: Unexpected error for user {user_id}: {e}")
        sys.stdout.flush()
        traceback.print_exc()
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail="Failed to fetch user profile data due to an internal error.")