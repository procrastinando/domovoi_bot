import os
import yaml
import logging
from config import DATABASE_FILE

logger = logging.getLogger(__name__)

def load_database():
    if not os.path.exists(DATABASE_FILE):
        return {}
    try:
        with open(DATABASE_FILE, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Error loading database: {e}")
        return {}

def save_database(data):
    try:
        with open(DATABASE_FILE, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        logger.error(f"Error saving database: {e}")

def get_user_data(user_id, key, default=None):
    return load_database().get(str(user_id), {}).get(key, default)

def set_user_data(user_id, key, value):
    db = load_database()
    db.setdefault(str(user_id), {})[key] = value
    save_database(db)