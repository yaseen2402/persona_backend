# app/api/routes_chat.py
from fastapi import APIRouter, HTTPException, Header
from typing import Optional, List, Dict, Any
import sys
import traceback
import json # For literal_eval if needed, or just for general json

from app.models.pydantic_models import ChatRequest, ChatResponse, MessageLimitErrorResponse
from app.core.security import verify_jwt_token
from app.services.llm_service import llm # Shared LLM client
from app.services.supabase_client import supabase # Shared Supabase client
from app.services.embedding_service import get_embedding
from app.services.persona_service import get_ai_persona
from app.services.user_service import check_message_limits_with_plan, get_user_profile_stats_logic # User service for limits
from app.tasks.analysis_tasks import perform_analysis_task
from app.core.config import USER_PERSONA_STATES_TABLE, MESSAGES_TABLE, USER_MESSAGE_STATS_TABLE
from app.tasks.post_chat_processing_tasks import save_ai_message_task, update_message_stats_task

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

router = APIRouter()

# This can be a constant if it never changes
ADDITIONAL_LLM_PROMPT = """
You are given a history of your past conversation with the user and a message user has asked you now, 
if history is empty or irrelevant ignore it and answer user's question without using it. 
Otherwise use history as context and answer the user's question.
"""

@router.post("/chat", response_model=ChatResponse, responses={403: {"model": MessageLimitErrorResponse}})
async def chat_route(
    chat_data: ChatRequest,
    authorization: Optional[str] = Header(None)
):
    print(f"[API] POST /chat called with ai_id: {chat_data.ai_id}")
    sys.stdout.flush()

    if not llm:
        raise HTTPException(status_code=503, detail="AI model service not available.")
    if not supabase:
        raise HTTPException(status_code=503, detail="Database service not available.")

    user_id = verify_jwt_token(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing token")
    
    print(f"[API] /chat: User {user_id} authenticated.")
    sys.stdout.flush()
    user_profile_data = get_user_profile_stats_logic(user_id)
    if user_profile_data is None:
        # This means a critical error occurred fetching profile/plan data
        raise HTTPException(status_code=503, detail="Service temporarily unavailable (user data fetch failed).")
    

    can_send, limit_error_msg = check_message_limits_with_plan(
        current_daily_count=user_profile_data['daily_message_count'],
        current_total_count=user_profile_data['total_message_count'],
        plan_daily_limit=user_profile_data['daily_message_limit'],
        plan_total_limit=user_profile_data['total_message_limit'],
        user_id=user_id
    )

    if not can_send:
        if limit_error_msg: # Ensure there's a message
             print(f"[API] /chat: User {user_id} limit check failed: {limit_error_msg}")
             sys.stdout.flush()
             raise HTTPException(status_code=403, detail=limit_error_msg)
        else: # Should not happen if check_message_limits_with_plan returns a message on False
             raise HTTPException(status_code=403, detail="Message limit exceeded.")

    # If can_send is True, stats_or_error_msg is a dict
    daily_count = user_profile_data['daily_message_count'] # This is the count *before* the current message
    total_count = user_profile_data['total_message_count']

    ai_id = chat_data.ai_id
    user_prompt = chat_data.prompt
    frontend_history = chat_data.history

    print(f"[API] /chat: AI '{ai_id}', Prompt: '{user_prompt[:50]}...'")
    sys.stdout.flush()

    persona = get_ai_persona(ai_id)
    if not persona:
        raise HTTPException(status_code=404, detail=f'AI persona with ID "{ai_id}" not found')

    user_persona_state = None
    try:
        # Fetch user persona state (using maybe_single for robustness)
        state_response = supabase.table(USER_PERSONA_STATES_TABLE) \
            .select('*') \
            .eq('user_id', str(user_id)) \
            .eq('persona_id', ai_id) \
            .maybe_single() \
            .execute()

        if state_response.data:
            user_persona_state = state_response.data
            print(f"[API] /chat: Fetched persona state for user {user_id}, AI {ai_id}.")
            sys.stdout.flush()
        else:
            print(f"[API] /chat: No persona state for user {user_id}, AI {ai_id}. Creating default.")
            sys.stdout.flush()
            default_state = {
                'user_id': str(user_id), 'persona_id': ai_id, 'attachment_level': 1, 
                'trust_level': 1, 'mood': 'neutral', 'introversion_level': 1, 
                'relationship_stage': 'stranger', 'openness': 1, 'energy_level': 1, 
                'user_summary': '', 'user_facts': {}
            }
            # Insert and then use the inserted data (or default if insert fails)
            insert_response = supabase.table(USER_PERSONA_STATES_TABLE).insert(default_state).execute()
            if insert_response.data and len(insert_response.data) > 0:
                user_persona_state = insert_response.data[0]
                print(f"[API] /chat: Created and using new persona state.")
                sys.stdout.flush()
            else: # Fallback if insert fails
                user_persona_state = default_state 
                print(f"[API] /chat: Used local default state after insert attempt. Insert error (if any): {getattr(insert_response, 'error', 'N/A')}")
                sys.stdout.flush()
    except Exception as e:
        print(f"[API] /chat: Error fetching/creating persona state: {e}")
        sys.stdout.flush()
        traceback.print_exc()
        sys.stdout.flush()
        # Decide if to proceed with a basic default or error out
        user_persona_state = { # Basic default for context
            'mood': 'neutral', 'relationship_stage': 'stranger', 'user_summary': '', 'user_facts': {}
        }


    # 1) Embed & save user message
    user_emb = get_embedding(user_prompt)
    if user_emb is None:
        raise HTTPException(status_code=500, detail="Failed to generate embedding for user prompt.")
    
    try:
        supabase.table(MESSAGES_TABLE).insert([{
            'user_id': str(user_id), 'ai_id': ai_id, 'content': user_prompt,
            'is_from_user': True, 'embedding': user_emb
        }]).execute()
        print(f"[API] /chat: User message saved.")
        sys.stdout.flush()
    except Exception as e:
        print(f"[API] /chat: Error saving user message: {e}")
        sys.stdout.flush() # Log and continue, or raise error if critical


    # 2) Find relevant context messages
    context_messages_for_llm = []
    try:
        # Using the RPC function for semantic search
        relevant_messages_response = supabase.rpc('match_messages_by_embedding', {
            'input_user_id': str(user_id),
            'input_ai_id': ai_id,
            'query_embedding': user_emb,
            'match_count': 5 # Get a few relevant messages
        }).execute()
        if relevant_messages_response.data:
            context_messages_for_llm = relevant_messages_response.data
            print(f"[API] /chat: Retrieved {len(context_messages_for_llm)} context messages via RPC.")
            sys.stdout.flush()
    except Exception as e:
        print(f"[API] /chat: Error fetching context messages via RPC: {e}")
        sys.stdout.flush()
        # Continue without this specific context if it fails

    # 3) Build messages for LLM
    chat_msgs_for_llm = [SystemMessage(content=ADDITIONAL_LLM_PROMPT)]
    chat_msgs_for_llm.append(SystemMessage(content=persona['instruction']))

    if user_persona_state: # Ensure it's not None
        state_context_str = f"""
        User's current state with this persona (ID: {ai_id}):
        - Attachment Level: {user_persona_state.get('attachment_level', 'N/A')}
        - Trust Level: {user_persona_state.get('trust_level', 'N/A')}
        - Mood: {user_persona_state.get('mood', 'N/A')}
        - User Summary: {user_persona_state.get('user_summary', 'N/A')}
        - User Facts: {json.dumps(user_persona_state.get('user_facts', {}))}
        Use this information to inform your response and maintain context.
        """
        chat_msgs_for_llm.append(SystemMessage(content=state_context_str))

    # Add DB-fetched context messages (semantically similar)
    if context_messages_for_llm:
        chat_msgs_for_llm.append(SystemMessage(content="Some relevant past conversation snippets:"))
        for msg_ctx in sorted(context_messages_for_llm, key=lambda x: x.get('created_at', '')): # Sort by time
            role_ctx = 'user' if msg_ctx.get('is_from_user') else ai_id
            content_ctx = msg_ctx.get('content', '')
            if role_ctx == 'user':
                chat_msgs_for_llm.append(HumanMessage(content=content_ctx))
            else:
                chat_msgs_for_llm.append(AIMessage(content=content_ctx, role=role_ctx))
    
    # Add frontend history (most recent turn-by-turn)
    if frontend_history:
        chat_msgs_for_llm.append(SystemMessage(content="Most Recent Conversation History (from client):"))
        for msg_hist in frontend_history:
            if 'text' in msg_hist and 'sender' in msg_hist:
                role_hist = 'user' if msg_hist['sender'] == 'user' else ai_id
                content_hist = msg_hist['text']
                if role_hist == 'user':
                    chat_msgs_for_llm.append(HumanMessage(content=content_hist))
                else:
                    chat_msgs_for_llm.append(AIMessage(content=content_hist, role=role_hist))

    chat_msgs_for_llm.append(SystemMessage(content="Current User Prompt:"))
    chat_msgs_for_llm.append(HumanMessage(content=user_prompt))
    
    print(f"[API] /chat: Sending {len(chat_msgs_for_llm)} message parts to LLM.")
    sys.stdout.flush()

    try:
        llm_response_obj = llm.invoke(chat_msgs_for_llm)
        llm_resp_content = llm_response_obj.content
        print(f"[API] /chat: LLM response received: {llm_resp_content[:100]}...")
        sys.stdout.flush()
    except Exception as e:
        print(f"[API] /chat: !!! EXCEPTION DURING llm.invoke() !!! : {e}")
        sys.stdout.flush()
        traceback.print_exc()
        sys.stdout.flush()
        raise HTTPException(status_code=503, detail="AI service failed to generate a response.")

    # 6) Save AI response
    ai_emb = get_embedding(llm_resp_content)
    if ai_emb is None:
        print(f"[API] /chat: Warning - Failed to generate embedding for AI response. Skipping save of embedding.")
        sys.stdout.flush()
        # Decide if this is critical enough to error out
    
    try:
        supabase.table(MESSAGES_TABLE).insert([{
            'user_id': str(user_id), 'ai_id': ai_id, 'content': llm_resp_content,
            'is_from_user': False, 'embedding': ai_emb # ai_emb can be None here
        }]).execute()
        print(f"[API] /chat: AI response saved.")
        sys.stdout.flush()
    except Exception as e:
        print(f"[API] /chat: Error saving AI response: {e}") # Log and continue
        sys.stdout.flush()


    # 7) Update message stats
    try:
        plan_daily_limit = user_profile_data.get('daily_message_limit') # Returns None if key missing or value is None
        plan_total_limit = user_profile_data.get('total_message_limit')
         # 1. Save AI message and its embedding via Celery
        print(f"[API] /chat: Queuing save_ai_message_task for user {user_id}.")
        sys.stdout.flush()
        save_ai_message_task.delay(
            user_id=str(user_id),
            ai_id=ai_id,
            llm_resp_content=llm_resp_content
        )

        # 2. Update message stats via Celery
        print(f"[API] /chat: Queuing update_message_stats_task for user {user_id}.")
        sys.stdout.flush()
        update_message_stats_task.delay(
            user_id=str(user_id),
            daily_count_before=daily_count,
            total_count_before=total_count,
            plan_daily_limit=plan_daily_limit,
            plan_total_limit=plan_total_limit
        )
        print(f"[API] /chat: Queuing perform_analysis_task for user {user_id}, AI {ai_id}.")
        sys.stdout.flush()
        perform_analysis_task.delay(str(user_id), ai_id, user_prompt)

    except Exception as e:
        print(f"[API] /chat: Error updating message stats: {e}") # Log and continue
        sys.stdout.flush()
    
    return ChatResponse(response=llm_resp_content)