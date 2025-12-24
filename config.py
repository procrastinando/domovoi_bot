import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DATABASE_FILE = "database.yaml"
LANGUAGE_FILE = "languages.yaml"
TELEGRAM_MSG_LIMIT = 4096

# Model Definitions with explicit Context Limits
MODELS = {
    "llama4": {
        "id": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "name": "Llama 4 Maverick",
        "type": "multimodal",
        "context": 8192
    },
    "kimi": {
        "id": "moonshotai/kimi-k2-instruct-0905",
        "name": "Kimi K2",
        "type": "text",
        "context": 16384
    },
    "gptoss": {
        "id": "openai/gpt-oss-120b",
        "name": "GPT-Oss 120B",
        "type": "reasoner",
        "context": 65536
    },
    "compound": {
        "id": "groq/compound",
        "name": "Compound",
        "type": "compound",
        "context": 8192
    }
}

DEFAULT_MODEL_KEY = "llama4"
DEFAULT_MAX_TOKENS = 4096

# Supported Languages
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Español',
    'zh': '中文',
    'ru': 'Русский',
    'fr': 'Français',
    'it': 'Italiano',
    'pt': 'Português',
    'vi': 'Tiếng Việt',
    'id': 'Bahasa Indonesia'
}

TAVILY_API_URL = "https://api.tavily.com/search"
TAVILY_USAGE_URL = "https://api.tavily.com/usage"