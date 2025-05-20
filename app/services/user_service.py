# app/services/user_service.py
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import sys
import traceback

from app.services.supabase_client import supabase # Shared Supabase client
from app.core.config import (
    USER_MESSAGE_STATS_TABLE,
    USER_SUBSCRIPTIONS_TABLE,
    SUBSCRIPTION_PLAN_TABLE
)

# Note: This service uses the shared 'supabase' client instance.
# If a Celery task needed these functions, it should pass its own task-specific client.
def check_message_limits_with_plan(
    current_daily_count: int,
    current_total_count: int,
    plan_daily_limit: Optional[int], # Can be None for unlimited
    plan_total_limit: Optional[int], # Can be None for unlimited
    user_id: str # For logging
) -> Tuple[bool, Optional[str]]:
    """
    Checks message limits against provided plan limits.
    Returns: (can_send: bool, error_message: Optional[str])
    """
    print(f"[user_service.check_message_limits_with_plan] User {user_id}: Counts D:{current_daily_count}, T:{current_total_count}. Limits D:{plan_daily_limit}, T:{plan_total_limit}")
    sys.stdout.flush()

    # Check daily limit
    if plan_daily_limit is not None and current_daily_count >= plan_daily_limit:
        msg = f"Daily message limit ({plan_daily_limit}) exceeded."
        print(f"[user_service.check_message_limits_with_plan] User {user_id}: {msg}")
        sys.stdout.flush()
        return False, msg
    
    # Check total limit
    if plan_total_limit is not None and current_total_count >= plan_total_limit:
        msg = f"Total message limit ({plan_total_limit}) exceeded."
        print(f"[user_service.check_message_limits_with_plan] User {user_id}: {msg}")
        sys.stdout.flush()
        return False, msg

    return True, None # No error message means limits are okay



def get_user_profile_stats_logic(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Contains the core logic for fetching user profile stats.
    Returns a dictionary of stats or None if a critical error occurs.
    """
    if not supabase:
        print("[user_service.get_user_profile_stats_logic] Supabase client not available.")
        sys.stdout.flush()
        return None

    try:
        # 1. Get User Message Stats (using maybe_single)
        stats_response = supabase.table(USER_MESSAGE_STATS_TABLE) \
            .select('daily_count, total_count, last_reset_at') \
            .eq('user_id', user_id) \
            .single() \
            .execute()

        daily_count = 0
        total_count = 0
        last_reset_at = None
        if stats_response.data:
            stats_data = stats_response.data
            daily_count = stats_data.get('daily_count', 0)
            total_count = stats_data.get('total_count', 0)
            last_reset_at = stats_data.get('last_reset_at')
        else:
            # Optionally create user_message_stats record if it doesn't exist
            print(f"[user_service] No message stats found for {user_id}, assuming 0 counts.")
            sys.stdout.flush()


        # 2. Determine Plan and Get Limits
        plan_name = 'free' # Default
        daily_limit: Optional[int] = 20 # Default free limits
        total_limit: Optional[int] = 500 # Default free limits

        # Query for active subscription (using maybe_single)
        subscription_response = supabase.table(USER_SUBSCRIPTIONS_TABLE) \
            .select('plan_id') \
            .eq('user_id', user_id) \
            .eq('is_active', True) \
            .single() \
            .execute()

        if subscription_response.data and 'plan_id' in subscription_response.data:
            plan_id = subscription_response.data['plan_id']
            plan_response = supabase.table(SUBSCRIPTION_PLAN_TABLE) \
                .select('name, daily_limit, total_limit') \
                .eq('id', plan_id) \
                .single() \
                .execute()

            if plan_response.data:
                plan_data = plan_response.data
                plan_name = plan_data.get('name', 'Unknown Paid Plan')
                daily_limit = plan_data.get('daily_limit') # Allow None for unlimited
                total_limit = plan_data.get('total_limit') # Allow None for unlimited
                print(f"[user_service] User {user_id} on plan: {plan_name}")
                sys.stdout.flush()
            else:
                print(f"[user_service] Plan details not found for plan_id: {plan_id}. Defaulting to free.")
                sys.stdout.flush()
                # Fallback to free plan limits if subscribed plan details are missing (should not happen ideally)
                # This logic for fetching 'free' plan details can be a shared utility
                free_plan_details = supabase.table(SUBSCRIPTION_PLAN_TABLE).select('daily_limit, total_limit').eq('name', 'free').maybe_single().execute()
                if free_plan_details.data:
                    daily_limit = free_plan_details.data.get('daily_limit', 20)
                    total_limit = free_plan_details.data.get('total_limit', 500)
                plan_name = 'free (paid plan details missing)'
        else:
            # No active paid subscription, ensure free plan limits are fetched/applied
            print(f"[user_service] User {user_id} no active paid subscription. Applying free plan limits.")
            sys.stdout.flush()
            free_plan_details = supabase.table(SUBSCRIPTION_PLAN_TABLE).select('daily_limit, total_limit').eq('name', 'free').maybe_single().execute()
            if free_plan_details.data:
                plan_name = 'free'
                daily_limit = free_plan_details.data.get('daily_limit', 20)
                total_limit = free_plan_details.data.get('total_limit', 500)
            else:
                print(f"[user_service] 'free' plan details not found in DB! Using hardcoded defaults.")
                sys.stdout.flush()
                plan_name = 'free (DB details missing)'
                # daily_limit, total_limit remain hardcoded defaults

        return {
            'plan_type': plan_name,
            'daily_message_count': daily_count,
            'daily_message_limit': daily_limit,
            'total_message_count': total_count,
            'total_message_limit': total_limit,
            'last_reset_at': last_reset_at
        }

    except Exception as e:
        print(f"[user_service.get_user_profile_stats_logic] Unexpected error for user {user_id}: {e}")
        sys.stdout.flush()
        traceback.print_exc()
        sys.stdout.flush()
        return None