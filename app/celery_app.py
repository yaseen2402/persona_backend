# app/celery_app.py
from celery import Celery
from app.core.config import REDIS_URL

# The first argument to Celery is the name of the current module.
# This is needed so that names can be generated automatically when tasks are defined in that module.
# Or a more descriptive name like "ai_chat_worker"
celery_app = Celery(
    "app", # Using 'app' as the main package name for tasks
    broker=REDIS_URL,
    include=[
        'app.tasks.analysis_tasks',
        'app.tasks.post_chat_processing_tasks'] # Auto-discover tasks from this module
)

# Optional Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # Example: If you want to add broker connection retry settings
    broker_connection_retry_on_startup=True,
    # broker_transport_options = { # Example for SSL, often not needed if rediss:// handles it
    #     'ssl_cert_reqs': 'CERT_NONE', # Adjust for your security requirements
    # }
)

if __name__ == '__main__':
    # This allows running the worker directly using: python -m app.celery_app worker -l info
    celery_app.start()