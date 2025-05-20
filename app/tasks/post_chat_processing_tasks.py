# app/tasks/post_chat_processing_tasks.py
import sys
import traceback
from typing import Optional, List, Any # For type hints if needed for embedding

from app.celery_app import celery_app
from app.services.supabase_client import get_new_supabase_client_for_task # Use task-specific client
from app.services.embedding_service import get_embedding # This can be called from the task
from app.core.config import MESSAGES_TABLE, USER_MESSAGE_STATS_TABLE

# Note: We pass data to these tasks, they don't re-fetch from main app context.

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60) # Retry after 1 min
def save_ai_message_task(
    self, # for bind=True, allows self.retry
    user_id: str,
    ai_id: str,
    llm_resp_content: str
):
    print(f"[Celery Task - save_ai_message_task] START for user {user_id}, AI {ai_id}")
    sys.stdout.flush()

    task_supabase = get_new_supabase_client_for_task()
    if not task_supabase:
        print("[Celery Task - save_ai_message_task] Supabase client init failed.")
        sys.stdout.flush()
        raise self.retry(exc=Exception("Supabase client init failed in save_ai_message_task"))

    ai_emb = None
    try:
        ai_emb = get_embedding(llm_resp_content) # get_embedding uses its own shared model instance
        if ai_emb is None:
            print(f"[Celery Task - save_ai_message_task] Warning: Failed to generate embedding for AI response.")
            sys.stdout.flush()
    except Exception as e:
        print(f"[Celery Task - save_ai_message_task] Error generating embedding: {e}")
        sys.stdout.flush()
        # Continue, will save message with null embedding

    try:
        insert_response = task_supabase.table(MESSAGES_TABLE).insert([{
            'user_id': str(user_id), 
            'ai_id': ai_id, 
            'content': llm_resp_content,
            'is_from_user': False, 
            'embedding': ai_emb # ai_emb can be None
        }]).execute()

        if getattr(insert_response, 'error', None):
            print(f"[Celery Task - save_ai_message_task] Error saving AI response: {insert_response.error}")
            sys.stdout.flush()
            # Consider retrying for specific DB errors
            raise Exception(f"DB error saving AI message: {insert_response.error}")
        else:
            print(f"[Celery Task - save_ai_message_task] AI response saved successfully.")
            sys.stdout.flush()
        return {"status": "success", "message": "AI message saved."}
    except Exception as e:
        print(f"[Celery Task - save_ai_message_task] Unexpected error: {e}")
        sys.stdout.flush()
        traceback.print_exc()
        sys.stdout.flush()
        raise self.retry(exc=e) # Default retry behavior

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def update_message_stats_task(
    self, # for bind=True
    user_id: str,
    # Pass the counts *before* the current message and the plan limits
    daily_count_before: int,
    total_count_before: int,
    plan_daily_limit: Optional[int],
    plan_total_limit: Optional[int]
):
    print(f"[Celery Task - update_message_stats_task] START for user {user_id}")
    sys.stdout.flush()

    task_supabase = get_new_supabase_client_for_task()
    if not task_supabase:
        print("[Celery Task - update_message_stats_task] Supabase client init failed.")
        sys.stdout.flush()
        raise self.retry(exc=Exception("Supabase client init failed in update_message_stats_task"))
    
    try:
        new_daily_count = daily_count_before + 1
        new_total_count = total_count_before + 1
        stats_update_payload = {}

        if plan_daily_limit is None or new_daily_count <= plan_daily_limit:
            stats_update_payload['daily_count'] = new_daily_count
        elif plan_daily_limit is not None: # Cap at limit
            stats_update_payload['daily_count'] = plan_daily_limit
        
        if plan_total_limit is None or new_total_count <= plan_total_limit:
            stats_update_payload['total_count'] = new_total_count
        elif plan_total_limit is not None: # Cap at limit
            stats_update_payload['total_count'] = plan_total_limit

        if stats_update_payload:
            print(f"[Celery Task - update_message_stats_task] Updating stats for user {user_id} with: {stats_update_payload}")
            sys.stdout.flush()
            update_response = task_supabase.table(USER_MESSAGE_STATS_TABLE) \
                .update(stats_update_payload) \
                .eq('user_id', str(user_id)) \
                .execute()
            
            if getattr(update_response, 'error', None):
                print(f"[Celery Task - update_message_stats_task] Error updating stats: {update_response.error}")
                sys.stdout.flush()
                raise Exception(f"DB error updating stats: {update_response.error}")
            else:
                print(f"[Celery Task - update_message_stats_task] Stats updated successfully.")
                sys.stdout.flush()
            return {"status": "success", "message": "Message stats updated."}
        else:
            print(f"[Celery Task - update_message_stats_task] No stat changes to update.")
            sys.stdout.flush()
            return {"status": "success", "message": "No stat changes needed."}
    except Exception as e:
        print(f"[Celery Task - update_message_stats_task] Unexpected error: {e}")
        sys.stdout.flush()
        traceback.print_exc()
        sys.stdout.flush()
        raise self.retry(exc=e)