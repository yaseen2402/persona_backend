# app/tasks/analysis_tasks.py
import json
import traceback
import sys
import re # For improved JSON parsing

from app.celery_app import celery_app
# Use the specific "get_new_..._for_task" client getters
from app.services.supabase_client import get_new_supabase_client_for_task
from app.services.llm_service import get_new_llm_client_for_task
from app.core.config import USER_PERSONA_STATES_TABLE

from langchain_core.messages import HumanMessage 
# Removed SystemMessage, AIMessage as they are not directly used in this task's LLM call's input list

@celery_app.task(bind=True, max_retries=3) # bind=True allows access to self for retries
def perform_analysis_task(self, user_id_str: str, ai_id: str, user_prompt: str):
    print(f"[Celery Task - START] perform_analysis_task for user {user_id_str}, AI {ai_id}.")
    sys.stdout.flush()

    task_supabase = get_new_supabase_client_for_task()
    task_llm = get_new_llm_client_for_task()

    if task_supabase is None:
        print("[Celery Task] Supabase client init failed for task.")
        sys.stdout.flush()
        raise self.retry(exc=Exception("Supabase client init failed in task"), countdown=60) # Retry after 60s
    
    if task_llm is None:
        print("[Celery Task] LLM client init failed for task.")
        sys.stdout.flush()
        raise self.retry(exc=Exception("LLM client init failed in task"), countdown=60)

    try:
        user_persona_state = None
        default_state_structure = {
            'user_id': user_id_str, 'persona_id': ai_id,
            'attachment_level': 1, 'trust_level': 1, 'mood': 'neutral',
            'introversion_level': 3, 'relationship_stage': 'stranger',
            'openness': 3, 'energy_level': 3, 'user_summary': '', 'user_facts': {}
        }
        try:
            print(f"[Celery Task] Fetching current user persona state.")
            sys.stdout.flush()
            state_response = task_supabase.table(USER_PERSONA_STATES_TABLE) \
                .select('*') \
                .eq('user_id', user_id_str) \
                .eq('persona_id', ai_id) \
                .maybe_single() \
                .execute()

            if state_response.data:
                user_persona_state = state_response.data
                print(f"[Celery Task] Successfully fetched state: {user_persona_state.get('mood')}") # Example log
            else:
                print(f"[Celery Task] No existing state found. Using default structure.")
                user_persona_state = default_state_structure
            sys.stdout.flush()
        except Exception as e:
            print(f"[Celery Task] Error fetching user persona state: {e}")
            sys.stdout.flush()
            traceback.print_exc()
            sys.stdout.flush()
            user_persona_state = default_state_structure # Fallback
            print(f"[Celery Task] Using default state due to fetch error.")
            sys.stdout.flush()

        analysis_prompt_text = f"""
        Analyze the following user prompt in the context of the user's current relationship and state with the AI persona (ID: {ai_id}).
        Based on the user's prompt, determine if any of the following attributes of the user's state should be updated.
        Provide the suggested updates in JSON format. Only include the attributes that need to be updated.
        For numerical attributes (attachment_level, trust_level, introversion_level, openness, energy_level), suggest an increment ('+1'), a decrement ('-1'), or a specific new integer value (1-10).
        For text attributes (mood, relationship_stage, user_summary), provide the new string value.
        For user_facts (jsonb), provide a JSON object with key-value pairs to add or update existing facts or add new ones.
        
        Current User State:
        {json.dumps(user_persona_state, indent=2)}
        
        User Prompt: "{user_prompt}"
        
        Suggested Updates (JSON format, only include attributes to update):
        """
        print(f"[Celery Task] Sending analysis prompt to LLM (length {len(analysis_prompt_text)}).")
        sys.stdout.flush()
        
        analysis_llm_response_content = task_llm.invoke([HumanMessage(content=analysis_prompt_text)]).content
        print(f"[Celery Task] Received LLM analysis (first 200 chars): {analysis_llm_response_content[:200]}...")
        sys.stdout.flush()

        updates = {}
        try:
            match = re.search(r"```json\s*([\s\S]*?)\s*```|({[\s\S]*})", analysis_llm_response_content)
            if match:
                json_string = match.group(1) or match.group(2)
                updates = json.loads(json_string)
                print(f"[Celery Task] Parsed JSON updates: {updates}")
            else:
                print(f"[Celery Task] No valid JSON block found in LLM response.")
            sys.stdout.flush()
        except json.JSONDecodeError as e:
            print(f"[Celery Task] Failed to parse JSON from LLM response: {e}")
            print(f"[Celery Task] LLM response was: {analysis_llm_response_content}")
            sys.stdout.flush()

        if updates:
            valid_columns = ['attachment_level', 'trust_level', 'mood', 'introversion_level',
                             'relationship_stage', 'openness', 'energy_level', 'user_summary', 'user_facts']
            filtered_llm_updates = {k: v for k, v in updates.items() if k in valid_columns}

            if filtered_llm_updates:
                print(f"[Celery Task] Applying LLM suggested updates: {filtered_llm_updates}")
                sys.stdout.flush()
                
                current_db_state_for_update_resp = task_supabase.table(USER_PERSONA_STATES_TABLE) \
                    .select('*').eq('user_id', user_id_str).eq('persona_id', ai_id).maybe_single().execute()
                
                current_db_state = current_db_state_for_update_resp.data if current_db_state_for_update_resp.data else default_state_structure
                
                final_updates_to_db = {}
                for key, value in filtered_llm_updates.items():
                    current_val_in_db = current_db_state.get(key)
                    if isinstance(value, str) and (value == '+1' or value == '-1'):
                        if isinstance(current_val_in_db, int):
                            new_val = current_val_in_db + (1 if value == '+1' else -1)
                            final_updates_to_db[key] = max(1, min(new_val, 10))
                    elif key == 'user_facts' and isinstance(value, dict):
                        existing_facts = current_db_state.get('user_facts', {}) if isinstance(current_db_state.get('user_facts'), dict) else {}
                        final_updates_to_db[key] = {**existing_facts, **value}
                    elif key in ['attachment_level', 'trust_level', 'introversion_level', 'openness', 'energy_level']:
                        try_val = None
                        if isinstance(value, int): try_val = value
                        elif isinstance(value, str) and value.lstrip('-').isdigit(): try_val = int(value)
                        if try_val is not None and 1 <= try_val <= 10:
                            final_updates_to_db[key] = try_val
                    elif key in ['mood', 'relationship_stage', 'user_summary'] and isinstance(value, str):
                        final_updates_to_db[key] = value

                if final_updates_to_db:
                    print(f"[Celery Task] Final updates for DB: {final_updates_to_db}")
                    sys.stdout.flush()
                    upsert_payload = {**final_updates_to_db, 'user_id': user_id_str, 'persona_id': ai_id}
                    
                    # IMPORTANT: Define a UNIQUE constraint on (user_id, persona_id) in your Supabase table
                    # for on_conflict to work reliably for upsert.
                    # If your PK is on (user_id, persona_id), it might infer.
                    # Otherwise, specify constraint name: on_conflict='your_user_id_persona_id_unique_constraint_name'
                    db_response = task_supabase.table(USER_PERSONA_STATES_TABLE) \
                        .upsert(upsert_payload, on_conflict='user_id,persona_id') \
                        .execute()

                    if getattr(db_response, 'error', None):
                        print(f"[Celery Task] Error upserting persona state: {db_response.error}")
                        sys.stdout.flush()
                        # Consider retry or specific error handling
                        return {"status": "error", "message": f"DB upsert error: {db_response.error}"}
                    else:
                        print(f"[Celery Task] Persona state upserted. Data: {db_response.data}")
                        sys.stdout.flush()
                        return {"status": "success", "updated_state_preview": db_response.data[0] if db_response.data else "Upsert OK"}
                else:
                    print(f"[Celery Task] No valid DB changes after filtering LLM updates.")
                    sys.stdout.flush()
                    return {"status": "success", "message": "No DB changes from LLM analysis."}
            else:
                print(f"[Celery Task] LLM updates empty after filtering valid columns.")
                sys.stdout.flush()
                return {"status": "success", "message": "No valid attributes in LLM suggestions."}
        else:
            print(f"[Celery Task] No updates from LLM (parsing failed or empty response).")
            sys.stdout.flush()
            return {"status": "success", "message": "No updates from LLM."}

    except Exception as e:
        print(f"[Celery Task - UNEXPECTED GLOBAL ERROR] perform_analysis_task: {e}")
        sys.stdout.flush()
        traceback.print_exc()
        sys.stdout.flush()
        # Retry for unexpected errors
        raise self.retry(exc=e, countdown=int(self.request.retries * 20) + 60) # Exponential backoff + initial
    finally:
        print(f"[Celery Task - END] perform_analysis_task for user {user_id_str}, AI {ai_id}.")
        sys.stdout.flush()