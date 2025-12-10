import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DATABASE_FILE = "database.yaml"
LANGUAGE_FILE = "language.yaml"
TELEGRAM_MSG_LIMIT = 4096

DEFAULT_MODEL = { "name": "meta-llama/llama-4-maverick-17b-128e-instruct", "temperature": 1.0 }
DEFAULT_MAX_TOKENS = 8192

MODELS = { "Kimi k2": "moonshotai/kimi-k2-instruct-0905", "📷 Llama 4 Maverick": "meta-llama/llama-4-maverick-17b-128e-instruct", "GPT-Oss 120B": "openai/gpt-oss-120b" }
MODEL_MAX_TOKENS = { "moonshotai/kimi-k2-instruct": 16384, "meta-llama/llama-4-maverick-17b-128e-instruct": 8192, "openai/gpt-oss-120b": 65536 }

VISION_MODELS = ["meta-llama/llama-4-maverick-17b-128e-instruct"]

SUPPORTED_LANGUAGES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'zh': 'Chinese',
    'pt': 'Portuguese', 'ru': 'Russian', 'it': 'Italian', 'hi': 'Hindi',
    'vi': 'Vietnamese', 'id': 'Indonesian'
}