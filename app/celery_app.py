# app/celery_app.py
from celery import Celery
from app.core.config import REDIS_URL


celery_app = Celery(
    "app", 
    broker=REDIS_URL,
    include=[
        'app.tasks.analysis_tasks',
        'app.tasks.post_chat_processing_tasks'] 
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    
)

if __name__ == '__main__':
    celery_app.start()