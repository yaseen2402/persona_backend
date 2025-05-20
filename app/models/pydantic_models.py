# app/models/pydantic_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime

class ChatRequest(BaseModel):
    ai_id: str
    prompt: str
    history: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

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
    id: Any # Could be int or str depending on your DB's ID type for messages
    user_id: str
    ai_id: str
    content: str
    created_at: datetime
    is_from_user: bool

class UserProfileStatsResponse(BaseModel):
    plan_type: str
    daily_message_count: int
    daily_message_limit: Optional[int] = None
    total_message_count: int
    total_message_limit: Optional[int] = None
    last_reset_at: Optional[datetime] = None