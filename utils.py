import random
import logging
import asyncio
import re
import httpx
from datetime import datetime
from telegram import Update
from telegram.ext import ContextTypes
from telegraph.aio import Telegraph
from markdown_it import MarkdownIt

from database import get_user_data, set_user_data
from localization import get_text
from config import TELEGRAM_MSG_LIMIT, TAVILY_USAGE_URL

# --- Custom Logger ---
class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_fmt = "%(asctime)s - %(message)s"
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logger(name="bot"):
    # Silence libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)
    return logger

logger = setup_logger()
telegraph = Telegraph()

def log_activity(user_id, action):
    logger.info(f"{user_id} - {action}")

# --- API & Text Utils ---

def get_random_api_key(user_id: int, key_type: str = 'groq') -> str:
    if key_type == 'groq':
        main = get_user_data(user_id, 'groq_api_key')
        fallback = get_user_data(user_id, 'groq_fallback_api_key')
    else:
        main = get_user_data(user_id, 'tavily_api_key')
        fallback = get_user_data(user_id, 'tavily_fallback_api_key')
    
    options = [k for k in [main, fallback] if k and k.strip()]
    return random.choice(options) if options else None

def mask_api_key(key: str) -> str:
    if not key or len(key) < 10: return "Not Set"
    return f"{key[:7]}...{key[-3:]}"

def strip_think_tags(text: str) -> str:
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()

def escape_html(text: str) -> str:
    if not text: return ""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def markdown_to_html_content(text):
    md = MarkdownIt()
    return md.render(text)

async def get_tavily_usage_stats(api_key: str) -> str:
    if not api_key: return "N/A"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(TAVILY_USAGE_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                if 'account' in data and 'plan_limit' in data['account']:
                    used = data['account']['plan_usage']
                    limit = data['account']['plan_limit']
                    return f"{used}/{limit}"
                elif 'key' in data:
                    used = data['key']['usage']
                    limit = data['key']['limit']
                    return f"{used}/{limit}"
    except Exception:
        pass
    return "Error"

# --- Messaging Utils ---

async def send_response(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    """Returns the sent Message object to capture ID."""
    user_id = update.effective_user.id
    telegraph_mode = get_user_data(user_id, 'use_telegraph', 'Never')
    
    use_telegraph = (telegraph_mode == 'Always') or (telegraph_mode == 'Long messages' and len(text) > TELEGRAM_MSG_LIMIT)

    if use_telegraph:
        try:
            token = get_user_data(user_id, 'telegraph_token')
            if not token:
                account = await telegraph.create_account(short_name=update.effective_user.first_name or "User")
                token = account['access_token']
                set_user_data(user_id, 'telegraph_token', token)
            
            user_telegraph = Telegraph(access_token=token)
            title = f"DomoBot - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            html = markdown_to_html_content(text)
            
            page = await user_telegraph.create_page(title=title, html_content=html)
            return await update.message.reply_text(f"{get_text(user_id, 'telegraph_link_msg')}: {page['url']}")
        except Exception as e:
            logger.error(f"Telegraph error: {e}")

    # Standard Split
    if len(text) > TELEGRAM_MSG_LIMIT:
        sent_msg = None
        for i in range(0, len(text), TELEGRAM_MSG_LIMIT):
            sent_msg = await update.message.reply_text(text[i:i + TELEGRAM_MSG_LIMIT])
        return sent_msg
    else:
        return await update.message.reply_text(text)

async def send_ephemeral(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, duration: int = 3):
    """Sends a message using HTML parse mode and deletes it after X seconds."""
    try:
        msg = await context.bot.send_message(chat_id=chat_id, text=text, parse_mode='HTML')
        await asyncio.sleep(duration)
        await msg.delete()
    except Exception:
        pass

async def delete_trigger_message(update: Update):
    """Instantly deletes the user's message."""
    try:
        await update.message.delete()
    except Exception:
        pass