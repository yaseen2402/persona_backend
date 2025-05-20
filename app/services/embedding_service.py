# app/services/embedding_service.py
from sentence_transformers import SentenceTransformer
from app.core.config import HF_HOME # Use configured HF_HOME
import os
import sys

# If HF_HOME is defined in config and not already in environment, set it.
# This helps ensure SentenceTransformer uses the intended cache path.
if HF_HOME and (os.getenv("HF_HOME") != HF_HOME):
    os.environ['HF_HOME'] = HF_HOME
    print(f"Set HF_HOME environment variable to: {HF_HOME}")
    sys.stdout.flush()

_embed_model_instance: SentenceTransformer | None = None

def get_embedding_model() -> SentenceTransformer | None:
    global _embed_model_instance
    if _embed_model_instance is None:
        try:
            print(f"Attempting to load SentenceTransformer model. Using HF_HOME: {os.getenv('HF_HOME')}")
            sys.stdout.flush()
            _embed_model_instance = SentenceTransformer("BAAI/bge-small-en-v1.5")
            print("SentenceTransformer model 'BAAI/bge-small-en-v1.5' loaded successfully.")
            sys.stdout.flush()
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            sys.stdout.flush()
            _embed_model_instance = None
    return _embed_model_instance

embedding_model: SentenceTransformer | None = get_embedding_model()

def get_embedding(text: str) -> list[float] | None:
    if embedding_model:
        try:
            emb = embedding_model.encode(text, normalize_embeddings=True)
            return emb.tolist()
        except Exception as e:
            print(f"Error getting embedding for text: {e}")
            sys.stdout.flush()
            return None
    else:
        print("Embedding model not loaded, cannot get embedding.")
        sys.stdout.flush()
        return None