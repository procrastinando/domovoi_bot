import yaml
import logging
from config import LANGUAGE_FILE, SUPPORTED_LANGUAGES
from bot.database import get_user_data

logger = logging.getLogger(__name__)

LANGUAGES_DATA = {}

def load_languages():
    global LANGUAGES_DATA
    try:
        with open(LANGUAGE_FILE, 'r', encoding='utf-8') as f:
            LANGUAGES_DATA = yaml.safe_load(f)
        logger.info(f"Loaded {len(LANGUAGES_DATA)} languages from {LANGUAGE_FILE}")
    except FileNotFoundError:
        logger.error(f"CRITICAL: {LANGUAGE_FILE} not found. Bot cannot run without language definitions.")
        exit()
    except yaml.YAMLError as e:
        logger.error(f"CRITICAL: Error parsing {LANGUAGE_FILE}. The file is not a valid YAML.")
        logger.error("This often happens with unescaped special characters (like '\') in strings.")
        logger.error("FIX: Use the literal block scalar '|' for multi-line prompts in your YAML file.")
        logger.error(f"YAML parser error: {e}")
        exit()
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the language file: {e}")
        exit()

def get_text(user_id: int, key: str, **kwargs) -> str:
    lang_code = get_user_data(user_id, 'language', 'en')
    lang_strings = LANGUAGES_DATA.get(lang_code, LANGUAGES_DATA.get('en', {}))
    template = lang_strings.get(key, f"_{key}_")
    return template.format(**kwargs)