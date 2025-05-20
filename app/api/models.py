# app/api/routes_models.py
from fastapi import APIRouter
from typing import List
import sys

from app.models.pydantic_models import ModelInfo
from app.services.persona_service import get_ai_persona, _AI_PERSONAS # Using the loaded personas

router = APIRouter()

@router.get("/models", response_model=List[ModelInfo])
async def list_ai_models_route():
    print(f"[API] GET /models called.")
    sys.stdout.flush()
    models_list = []
    # _AI_PERSONAS should be loaded by persona_service on import
    for _id, details in _AI_PERSONAS.items():
        models_list.append(ModelInfo(
            id=details.get('id'),
            name=details.get('name'),
            description=details.get('description'),
            avatar_initial=details.get('avatar_initial', ''), # Provide default
            profile_pic_url=details.get('profile_pic_url', '') # Provide default
        ))
    print(f"[API] /models: Returning {len(models_list)} models.")
    sys.stdout.flush()
    return models_list