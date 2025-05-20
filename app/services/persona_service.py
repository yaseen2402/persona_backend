# app/services/persona_service.py
import json
from pathlib import Path
from app.core.config import PERSONAS_PATH
from typing import Dict, Any, Optional
import sys

_AI_PERSONAS: Dict[str, Any] = {}

def load_personas():
    global _AI_PERSONAS
    if PERSONAS_PATH.exists():
        try:
            with open(PERSONAS_PATH, "r", encoding="utf-8") as f:
                persona_list = json.load(f)
                _AI_PERSONAS = {p["id"]: p for p in persona_list}
                print(f"Successfully loaded {len(_AI_PERSONAS)} personas.")
                sys.stdout.flush()
        except Exception as e:
            print(f"Error loading personas from {PERSONAS_PATH}: {e}")
            sys.stdout.flush()
            _AI_PERSONAS = {} # Ensure it's empty on error
    else:
        print(f"Warning: Personas file not found at {PERSONAS_PATH}. No personas loaded.")
        sys.stdout.flush()
        _AI_PERSONAS = {}

load_personas() # Load personas when this module is imported

def get_ai_persona(ai_id: str) -> Optional[Dict[str, Any]]:
    return _AI_PERSONAS.get(ai_id)