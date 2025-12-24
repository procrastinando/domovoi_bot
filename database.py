import os
import yaml
import logging
from config import DATABASE_FILE

logger = logging.getLogger("bot")

def load_database():
    if not os.path.exists(DATABASE_FILE):
        return {}
    try:
        with open(DATABASE_FILE, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"DB Load Error: {e}")
        return {}

def save_database(data):
    try:
        with open(DATABASE_FILE, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        logger.error(f"DB Save Error: {e}")

def get_user_data(user_id, key, default=None):
    db = load_database()
    return db.get(str(user_id), {}).get(key, default)

def set_user_data(user_id, key, value):
    db = load_database()
    if str(user_id) not in db:
        db[str(user_id)] = {}
    db[str(user_id)][key] = value
    save_database(db)

def delete_user_data(user_id):
    db = load_database()
    if str(user_id) in db:
        del db[str(user_id)]
        save_database(db)