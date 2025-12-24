import yaml
import logging
from config import LANGUAGE_FILE
from database import get_user_data

logger = logging.getLogger("bot")

LANGUAGES_DATA = {}

def load_languages():
    global LANGUAGES_DATA
    try:
        with open(LANGUAGE_FILE, 'r', encoding='utf-8') as f:
            LANGUAGES_DATA = yaml.safe_load(f)
    except Exception as e:
        print(f"CRITICAL: Error loading languages.yaml: {e}")
        exit()

def get_text(user_id: int, key: str, **kwargs) -> str:
    lang_code = get_user_data(user_id, 'language', 'en')
    lang_dict = LANGUAGES_DATA.get(lang_code, LANGUAGES_DATA.get('en', {}))
    text = lang_dict.get(key)
    
    if text is None:
        # Fallback to English
        text = LANGUAGES_DATA.get('en', {}).get(key, f"MISSING_KEY: {key}")
        
    try:
        return text.format(**kwargs)
    except Exception:
        return text