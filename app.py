import os
import logging
import base64
import yaml # Imported to catch specific YAML errors
import re
import asyncio
import subprocess
import json
from datetime import datetime, timezone
from io import BytesIO
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand, BotCommandScopeChat
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.error import BadRequest
from groq import Groq, RateLimitError, APIStatusError
from ddgs import DDGS
from telegraph.aio import Telegraph
from markdown_it import MarkdownIt

# --- Configuration ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DATABASE_FILE = "database.yaml"
LANGUAGE_FILE = "language.yaml"
TELEGRAM_MSG_LIMIT = 4096

DEFAULT_MODEL = { "name": "meta-llama/llama-4-maverick-17b-128e-instruct", "temperature": 1.0 }
DEFAULT_MAX_TOKENS = 8192

MODELS = { "Kimi k2": "moonshotai/kimi-k2-instruct", "📷 Llama 4 Maverick": "meta-llama/llama-4-maverick-17b-128e-instruct", "GPT-Oss 120B": "openai/gpt-oss-120b" }
MODEL_MAX_TOKENS = { "moonshotai/kimi-k2-instruct": 16384, "meta-llama/llama-4-maverick-17b-128e-instruct": 8192, "openai/gpt-oss-120b": 65536 }

# --- Logging Setup ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
telegraph = Telegraph()

# --- Language Data and Helper ---
LANGUAGES_DATA = {}
SUPPORTED_LANGUAGES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'zh': 'Chinese',
    'pt': 'Portuguese', 'ru': 'Russian', 'it': 'Italian', 'hi': 'Hindi',
    'vi': 'Vietnamese', 'id': 'Indonesian'
}

def load_languages():
    global LANGUAGES_DATA
    try:
        with open(LANGUAGE_FILE, 'r', encoding='utf-8') as f:
            LANGUAGES_DATA = yaml.safe_load(f)
        logger.info(f"Loaded {len(LANGUAGES_DATA)} languages from {LANGUAGE_FILE}")
    except FileNotFoundError:
        logger.error(f"CRITICAL: {LANGUAGE_FILE} not found. Bot cannot run without language definitions.")
        exit()
    # --- MODIFIED --- More specific error handling for YAML parsing issues
    except yaml.YAMLError as e:
        logger.error(f"CRITICAL: Error parsing {LANGUAGE_FILE}. The file is not a valid YAML.")
        logger.error("This often happens with unescaped special characters (like '\\') in strings.")
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

# --- Helper Functions ---
def escape_markdown_v2(text: str) -> str:
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

def strip_think_tags(text: str) -> str:
    text = text.strip()
    match = re.search(r"<think>.*?</think>(.*)", text, flags=re.DOTALL)
    if match and (answer := match.group(1).strip()):
        return answer
    match = re.search(r"<think>(.*?)(</think>|$)", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

def markdown_to_telegraph_nodes(markdown_text):
    md = MarkdownIt()
    tokens = md.parse(markdown_text)
    nodes = []
    token_iter = iter(tokens)

    def parse_inline(token):
        children = []
        if not token.children:
            return [token.content] if token.content else []
        for child in token.children:
            if child.type == 'text':
                children.append(child.content)
            elif child.type == 'code_inline':
                children.append({'tag': 'code', 'children': [child.content]})
            elif child.type == 'strong_open':
                strong_content = next((c.content for c in token.children if c.type == 'text'), '')
                children.append({'tag': 'strong', 'children': [strong_content]})
            elif child.type == 'link_open':
                link_text = child.children[0].content if child.children else ''
                children.append({'tag': 'a', 'attrs': {'href': child.attrs['href']}, 'children': [link_text]})
        return children

    def parse_token(token, token_iterator):
        if token.type == 'heading_open':
            tag = min(token.tag, 'h4')
            children = parse_inline(next(token_iterator))
            next(token_iterator)
            return {'tag': tag, 'children': children}
        elif token.type == 'paragraph_open':
            children = parse_inline(next(token_iterator))
            next(token_iterator)
            return {'tag': 'p', 'children': children} if children and children != [''] else None
        elif token.type == 'bullet_list_open':
            items = []
            while (t := next(token_iterator)).type != 'bullet_list_close':
                if t.type == 'list_item_open':
                    next(token_iterator)
                    items.append({'tag': 'li', 'children': parse_inline(next(token_iterator))})
                    next(token_iterator)
            return {'tag': 'ul', 'children': items}
        elif token.type == 'fence':
            return {'tag': 'pre', 'children': [token.content]}
        return None

    while True:
        try:
            token = next(token_iter)
            if node := parse_token(token, token_iter):
                nodes.append(node)
        except StopIteration:
            break
    return nodes

async def send_temp_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, duration: int = 3):
    try:
        msg = await context.bot.send_message(chat_id=chat_id, text=text)
        await asyncio.sleep(duration)
        await msg.delete()
    except Exception as e:
        logger.warning(f"Could not send or delete temporary message: {e}")

async def send_response(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    user_id = update.effective_user.id
    telegraph_mode = get_user_data(user_id, 'use_telegraph', 'Never')
    use_telegraph_now = (telegraph_mode == 'Always' or (telegraph_mode == 'Long messages' and len(text) > TELEGRAM_MSG_LIMIT))

    if use_telegraph_now:
        try:
            token = get_user_data(user_id, 'telegraph_token')
            if not token:
                account = await telegraph.create_account(short_name=update.effective_user.first_name)
                token = account['access_token']
                set_user_data(user_id, 'telegraph_token', token)
            user_telegraph = Telegraph(access_token=token)
            nodes = markdown_to_telegraph_nodes(text)
            title = f"Response - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
            response = await user_telegraph.create_page(title=title, content=nodes)
            await update.message.reply_text(response['url'])
            return
        except Exception as e:
            logger.error(f"Telegraph error for user {user_id}: {e}. Falling back to splitting.")

    if len(text) > TELEGRAM_MSG_LIMIT:
        for i in range(0, len(text), TELEGRAM_MSG_LIMIT):
            await update.message.reply_text(text[i:i + TELEGRAM_MSG_LIMIT])
    else:
        await update.message.reply_text(text)

def is_api_key_valid_sync(api_key: str) -> bool:
    if not api_key or not api_key.startswith("gsk_"):
        return False
    try:
        Groq(api_key=api_key, max_retries=0).models.list()
        return True
    except Exception as e:
        logger.warning(f"API key validation failed: {e}")
        return False

# --- Database Functions ---
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
            yaml.dump(data, f, default_flow_style=False)
    except Exception as e:
        logger.error(f"Error saving database: {e}")
def get_user_data(user_id, key, default=None):
    return load_database().get(str(user_id), {}).get(key, default)
def set_user_data(user_id, key, value):
    db = load_database()
    db.setdefault(str(user_id), {})[key] = value
    save_database(db)

# --- Groq-related Functions ---
async def get_groq_completion_with_retry(update: Update, context: ContextTypes.DEFAULT_TYPE, messages: list, model_config: dict, effective_tokens: int):
    user_id = update.effective_user.id
    attempt_count = 0
    max_attempts_with_swap = 2

    while True:
        api_key = get_user_data(user_id, 'groq_api_key')
        if not api_key:
            await update.message.reply_text(get_text(user_id, 'api_key_not_set'))
            raise ValueError("Main API key not found.")
        groq_client = Groq(api_key=api_key, max_retries=0)
        attempt_count += 1
        try:
            params = {
                "model": model_config['name'], "temperature": float(model_config.get('temperature', 1.0)),
                "max_tokens": effective_tokens, "messages": messages
            }
            if 'reasoning_effort' in model_config:
                params['reasoning_effort'] = model_config['reasoning_effort']
            return await asyncio.to_thread(groq_client.chat.completions.create, **params)
        except (RateLimitError, APIStatusError) as e:
            if isinstance(e, APIStatusError) and e.status_code == 413:
                await update.message.reply_text(get_text(user_id, 'context_too_long'))
                raise
            fallback_key = get_user_data(user_id, 'fallback_api_key')
            if fallback_key and attempt_count < max_attempts_with_swap:
                logger.warning(f"Main API key failed for user {user_id}. Swapping to fallback key.")
                await send_temp_message(context, user_id, get_text(user_id, 'api_key_fallback_attempt'), 2)
                set_user_data(user_id, 'groq_api_key', fallback_key)
                set_user_data(user_id, 'fallback_api_key', api_key)
                continue
            if isinstance(e, RateLimitError):
                retry_after = int(e.response.headers.get("retry-after", 60))
                logger.warning(f"All API keys failed rate limit for user {user_id}. Retrying after {retry_after}s.")
                await update.effective_chat.send_message(get_text(user_id, 'rate_limit_message', seconds=retry_after))
                await asyncio.sleep(retry_after)
                attempt_count = 0
            else:
                logger.error(f"An unexpected API status error occurred for user {user_id}: {e}")
                await update.message.reply_text(get_text(user_id, 'api_error'))
                raise
        except Exception as e:
            logger.error(f"An unexpected error occurred calling Groq API for user {user_id}: {e}")
            raise

async def prepare_messages_for_groq(user_id, history, model_name):
    default_prompt = get_text(user_id, 'default_system_prompt')
    system_prompt = get_user_data(user_id, 'system_prompt', default_prompt)
    
    groq_messages = [{"role": "system", "content": f"Today is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}. {system_prompt}"}]
    is_multimodal = model_name == MODELS["📷 Llama 4 Maverick"]
    for message in history:
        msg_copy = message.copy()
        if isinstance(msg_copy.get("content"), list):
            if not is_multimodal:
                msg_copy["content"] = " ".join([item.get("text", "") for item in msg_copy["content"] if item.get("type") == "text"])
            else:
                new_content = []
                for item in msg_copy["content"]:
                    if item.get("type") == "image_bytes":
                        b64 = base64.b64encode(item["bytes"]).decode('utf-8')
                        new_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                    else:
                        new_content.append(item)
                msg_copy["content"] = new_content
        msg_copy.pop("timestamp", None)
        groq_messages.append(msg_copy)
    return groq_messages

# --- Telegram Bot Command & Menu Setup ---
async def set_bot_commands_for_user(user_id: int, context: ContextTypes.DEFAULT_TYPE):
    model_config = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
    model_friendly_name = [name for name, id in MODELS.items() if id == model_config['name']][0]
    model_display = f"{model_friendly_name} ({model_config.get('reasoning_effort', '')})".strip()
    
    user_tokens = get_user_data(user_id, 'max_completion_tokens', DEFAULT_MAX_TOKENS)
    model_limit = MODEL_MAX_TOKENS.get(model_config['name'], DEFAULT_MAX_TOKENS)
    effective_tokens = min(user_tokens, model_limit)
    
    web_search_status = get_text(user_id, 'menu_web_search_on') if get_user_data(user_id, 'web_search_enabled', False) else get_text(user_id, 'menu_web_search_off')
    
    commands = [
        BotCommand("new_chat", get_text(user_id, 'menu_new_chat')),
        BotCommand("models", model_display),
        BotCommand("web_search", web_search_status),
        BotCommand("system_prompt", get_text(user_id, 'menu_system_prompt')),
        BotCommand("image_prompt", get_text(user_id, 'menu_image_prompt')),
        BotCommand("use_telegraph", get_text(user_id, 'menu_use_telegraph', mode=get_user_data(user_id, 'use_telegraph', 'Never'))),
        BotCommand("temperature", get_text(user_id, 'menu_temperature', temp=model_config.get('temperature', 1.0))),
        BotCommand("max_completion_tokens", get_text(user_id, 'menu_max_tokens', user_tokens=user_tokens, effective_tokens=effective_tokens)),
        BotCommand("language", get_text(user_id, 'menu_language')),
        BotCommand("api", get_text(user_id, 'menu_api')),
        BotCommand("fallback_api", get_text(user_id, 'menu_fallback_api')),
        BotCommand("erase_me", get_text(user_id, 'menu_erase_me'))
    ]
    await context.bot.set_my_commands(commands, scope=BotCommandScopeChat(chat_id=user_id))
    logger.info(f"Updated command menu for user {user_id}")

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if get_user_data(user_id, 'language') is None:
        user_lang = (update.effective_user.language_code or 'en').split('-')[0]
        lang_to_set = user_lang if user_lang in SUPPORTED_LANGUAGES else 'en'
        set_user_data(user_id, 'language', lang_to_set)
        set_user_data(user_id, 'system_prompt', get_text(user_id, 'default_system_prompt'))
        set_user_data(user_id, 'image_prompt', get_text(user_id, 'default_image_prompt'))

    set_user_data(user_id, 'state', 'chatting')
    await set_bot_commands_for_user(user_id, context)

    if not await asyncio.to_thread(is_api_key_valid_sync, get_user_data(user_id, 'groq_api_key')):
        msg = await update.effective_chat.send_message(get_text(user_id, 'welcome_prompt'))
        set_user_data(user_id, 'state', 'awaiting_api_key')
        set_user_data(user_id, 'last_bot_message_id', msg.message_id)
    else:
        await update.effective_chat.send_message(get_text(user_id, 'hello_message'))

async def new_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    set_user_data(user_id, "current_chat", [])
    logger.info(f"Erased chat history for user {user_id}")
    await update.message.reply_text(get_text(user_id, 'new_chat_started'))

async def generic_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, user_state: str, prompt_text: str):
    user_id = update.effective_user.id
    await update.message.delete()
    keyboard = [[InlineKeyboardButton(get_text(user_id, 'button_cancel'), callback_data="cancel_generic")]]
    msg = await update.effective_chat.send_message(prompt_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='MarkdownV2')
    set_user_data(user_id, 'state', user_state)
    set_user_data(user_id, 'last_bot_message_id', msg.message_id)

async def api_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    key = get_user_data(user_id, 'groq_api_key', 'Not set')
    masked = f"{key[:7]}...{key[-4:]}" if '...' not in key else 'Not set'
    prompt = get_text(user_id, 'api_key_prompt', masked_key=escape_markdown_v2(masked))
    await generic_command_handler(update, context, 'awaiting_api_key', prompt.replace('.', '\\.'))

async def fallback_api_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    key = get_user_data(user_id, 'fallback_api_key', 'Not set')
    masked = f"{key[:7]}...{key[-4:]}" if '...' not in key else 'Not set'
    prompt = get_text(user_id, 'fallback_api_key_prompt', masked_key=escape_markdown_v2(masked))
    await generic_command_handler(update, context, 'awaiting_fallback_api_key', prompt.replace('.', '\\.'))

async def system_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    prompt = get_user_data(user_id, 'system_prompt', get_text(user_id, 'default_system_prompt'))
    text = get_text(user_id, 'system_prompt_prompt', prompt=escape_markdown_v2(prompt))
    await generic_command_handler(update, context, 'awaiting_system_prompt', text.replace('.', '\\.'))

async def image_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    prompt = get_user_data(user_id, 'image_prompt', get_text(user_id, 'default_image_prompt'))
    text = get_text(user_id, 'image_prompt_prompt', prompt=escape_markdown_v2(prompt))
    await generic_command_handler(update, context, 'awaiting_image_prompt', text.replace('.', '\\.'))

async def temperature_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    temp = get_user_data(user_id, 'selected_model', DEFAULT_MODEL).get('temperature', 1.0)
    text = get_text(user_id, 'temperature_prompt', temp=temp)
    await generic_command_handler(update, context, 'awaiting_temperature', text.replace('.', '\\.'))

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.delete()
    keyboard = [[InlineKeyboardButton(name, callback_data=f"model_{model_id}") for name, model_id in MODELS.items()]]
    await update.effective_chat.send_message(get_text(user_id, 'models_prompt'), reply_markup=InlineKeyboardMarkup(keyboard))

async def web_search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    await update.message.delete()
    new_status = not get_user_data(user_id, 'web_search_enabled', False)
    set_user_data(user_id, 'web_search_enabled', new_status)
    status_text = get_text(user_id, 'web_search_on_status' if new_status else 'web_search_off_status')
    await send_temp_message(context, user_id, status_text)
    await set_bot_commands_for_user(user_id, context)

async def max_completion_tokens_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.delete()
    keyboard = [
        [InlineKeyboardButton(str(val), callback_data=f"tokens_{val}") for val in [512, 1024, 2048, 4096]],
        [InlineKeyboardButton(str(val), callback_data=f"tokens_{val}") for val in [8192, 16384, 32768, 65536]]
    ]
    await update.effective_chat.send_message(get_text(user_id, 'max_tokens_prompt'), reply_markup=InlineKeyboardMarkup(keyboard))

async def use_telegraph_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.delete()
    keyboard = [[InlineKeyboardButton(val, callback_data=f"telegraph_{val}") for val in ["Never", "Long messages", "Always"]]]
    await update.effective_chat.send_message(get_text(user_id, 'use_telegraph_prompt'), reply_markup=InlineKeyboardMarkup(keyboard))

async def erase_me_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.delete()
    keyboard = [[
        InlineKeyboardButton(get_text(user_id, 'button_yes_erase'), callback_data="erase_confirm_yes"),
        InlineKeyboardButton(get_text(user_id, 'button_cancel'), callback_data="erase_confirm_cancel")
    ]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.effective_chat.send_message(get_text(user_id, 'erase_me_prompt'), reply_markup=reply_markup)
    
async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.delete()
    buttons = [InlineKeyboardButton(name, callback_data=f"lang_{code}") for code, name in SUPPORTED_LANGUAGES.items()]
    keyboard = [buttons[i:i + 3] for i in range(0, len(buttons), 3)]
    keyboard.append([InlineKeyboardButton("🚫 " + get_text(user_id, 'button_cancel'), callback_data="cancel_generic")])
    await update.effective_chat.send_message("Please choose your language:", reply_markup=InlineKeyboardMarkup(keyboard))

async def perform_web_search(query: str):
    logger.info(f"Performing web search for: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        return "\n\n--- Web Search Results ---\n" + "\n\n".join([f"Title: {r.get('title', 'N/A')}\nSnippet: {r.get('body', 'N/A')}" for r in results]) if results else "No web search results found."
    except Exception as e:
        logger.error(f"Error during web search: {e}")
        return None

# --- Callback and Core Logic ---
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    data = query.data

    delete_message = True

    if data.startswith("model_"):
        model_id = data.replace("model_", "")
        if model_id == MODELS["GPT-Oss 120B"]:
            keyboard = [[InlineKeyboardButton(e.title(), callback_data=f"reasoning_{e}_{model_id}") for e in ["low", "medium", "high"]]]
            await query.edit_message_text(get_text(user_id, 'reasoning_prompt'), reply_markup=InlineKeyboardMarkup(keyboard))
            return
        else:
            config = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
            set_user_data(user_id, 'selected_model', {"name": model_id, "temperature": config.get('temperature', 1.0)})
    elif data.startswith("reasoning_"):
        _, effort, model_id = data.split("_", 2)
        config = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
        set_user_data(user_id, 'selected_model', {"name": model_id, "temperature": config.get('temperature', 1.0), "reasoning_effort": effort})
    elif data.startswith("tokens_"):
        set_user_data(user_id, 'max_completion_tokens', int(data.split("_")[1]))
    elif data.startswith("telegraph_"):
        set_user_data(user_id, 'use_telegraph', data.split("_")[1])
    elif data.startswith("lang_"):
        lang_code = data.split("_")[1]
        set_user_data(user_id, 'language', lang_code)
        await query.edit_message_text(get_text(user_id, 'language_updated'))
        await asyncio.sleep(2)
    elif data == "cancel_generic":
        set_user_data(user_id, 'state', 'chatting')
    elif data == "erase_confirm_yes":
        db = load_database()
        if str(user_id) in db:
            del db[str(user_id)]
            save_database(db)
            logger.info(f"Erased all data for user {user_id}")
            await query.edit_message_text(get_text(user_id, 'data_erased_confirm'))
            await start(query, context) # Re-run start to re-initialize user
        else:
            await query.edit_message_text(get_text(user_id, 'no_data_to_erase'))
        return
    elif data == "erase_confirm_cancel":
        await query.edit_message_text(get_text(user_id, 'operation_cancelled'))
        await asyncio.sleep(2)

    if delete_message:
        try: await query.message.delete()
        except: pass
    
    await set_bot_commands_for_user(user_id, context)

async def handle_user_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message: return
    user_id = update.message.from_user.id
    state = get_user_data(user_id, 'state')

    async def cleanup_and_set_state(new_state='chatting'):
        await update.message.delete()
        if last_msg_id := get_user_data(user_id, 'last_bot_message_id'):
            try: await context.bot.delete_message(chat_id=user_id, message_id=last_msg_id)
            except: pass
        set_user_data(user_id, 'state', new_state)

    state_map = {
        'awaiting_api_key': ('groq_api_key', 'api_key_updated'),
        'awaiting_fallback_api_key': ('fallback_api_key', 'fallback_api_key_updated'),
        'awaiting_system_prompt': ('system_prompt', 'system_prompt_updated'),
        'awaiting_image_prompt': ('image_prompt', 'image_prompt_updated')
    }

    if state in state_map:
        key, msg_key = state_map[state]
        value = update.message.text
        if 'api_key' in key and not await asyncio.to_thread(is_api_key_valid_sync, value):
            await cleanup_and_set_state(); await send_temp_message(context, user_id, get_text(user_id, 'api_key_invalid')); return
        set_user_data(user_id, key, value); await cleanup_and_set_state(); await send_temp_message(context, user_id, get_text(user_id, msg_key)); return
    elif state == 'awaiting_temperature':
        try:
            temp = float(update.message.text)
            if 0.0 <= temp <= 1.0:
                config = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
                config['temperature'] = temp; set_user_data(user_id, 'selected_model', config)
                await send_temp_message(context, user_id, get_text(user_id, 'temperature_updated', temp=temp))
            else: await send_temp_message(context, user_id, get_text(user_id, 'invalid_value'))
        except ValueError: await send_temp_message(context, user_id, get_text(user_id, 'invalid_format'))
        await cleanup_and_set_state(); await set_bot_commands_for_user(user_id, context); return

    if not await asyncio.to_thread(is_api_key_valid_sync, get_user_data(user_id, 'groq_api_key')):
        await start(update, context); return

    if update.message.voice or update.message.audio: await handle_audio_message(update, context)
    elif update.message.photo: await handle_photo_message(update, context)
    elif update.message.text: await handle_text_message(update, context)

async def process_and_respond(update: Update, context: ContextTypes.DEFAULT_TYPE, history: list):
    user_id = update.effective_user.id
    try:
        model_config = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
        user_tokens = get_user_data(user_id, 'max_completion_tokens', DEFAULT_MAX_TOKENS)
        model_limit = MODEL_MAX_TOKENS.get(model_config['name'], DEFAULT_MAX_TOKENS)
        effective_tokens = min(user_tokens, model_limit)
        messages = await prepare_messages_for_groq(user_id, history, model_config['name'])
        completion = await get_groq_completion_with_retry(update, context, messages, model_config, effective_tokens)
        response_text = strip_think_tags(completion.choices[0].message.content)

        if not response_text.strip():
            logger.warning(f"Model generated an empty response for user {user_id}. Model: {model_config.get('name')}")
            await update.message.reply_text(get_text(user_id, 'empty_response_error'))
            return

        history.append({"role": "assistant", "content": response_text, "timestamp": datetime.now(timezone.utc).isoformat()})
        set_user_data(user_id, 'current_chat', history)
        await send_response(update, context, response_text)
    except (APIStatusError, RateLimitError, ValueError):
        logger.warning(f"API call aborted for user {user_id} due to unrecoverable error.")
    except Exception as e:
        logger.error(f"Error in process_and_respond for user {user_id}: {e}", exc_info=True)
        await update.message.reply_text(get_text(user_id, 'internal_error'))

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    prompt = update.message.text
    if get_user_data(user_id, 'web_search_enabled', False):
        if results := await perform_web_search(prompt):
            prompt = f"{results}\n\nUser query: {prompt}"
        else:
            await send_temp_message(context, update.effective_chat.id, get_text(user_id, 'web_search_failed'))
    history = get_user_data(user_id, 'current_chat', [])
    history.append({"role": "user", "content": prompt, "timestamp": datetime.now(timezone.utc).isoformat()})
    await process_and_respond(update, context, history)

async def handle_audio_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    try:
        audio_file = await (update.message.voice or update.message.audio).get_file()
        audio_bytes_io = BytesIO()
        await audio_file.download_to_memory(out=audio_bytes_io)
        audio_bytes_io.seek(0)

        command = ['ffmpeg', '-i', 'pipe:0', '-f', 'mp3', '-y', 'pipe:1']
        process = await asyncio.create_subprocess_exec(
            *command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        converted_audio_bytes, stderr = await process.communicate(input=audio_bytes_io.read())
        if process.returncode != 0:
            raise IOError(f"FFmpeg failed: {stderr.decode()}")

        api_key = get_user_data(user_id, 'groq_api_key')
        transcription = await asyncio.to_thread(
            Groq(api_key=api_key, max_retries=0).audio.transcriptions.create,
            file=("audio.mp3", converted_audio_bytes), model="whisper-large-v3"
        )
        history = get_user_data(user_id, 'current_chat', [])
        history.append({"role": "user", "content": f'[Transcribed from audio]: "{transcription.text}"', "timestamp": datetime.now(timezone.utc).isoformat()})
        await process_and_respond(update, context, history)
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        await update.message.reply_text(get_text(user_id, 'audio_error'))

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    try:
        model_config = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
        photo_file = await update.message.photo[-1].get_file()
        
        photo_bytes = bytes(await photo_file.download_as_bytearray())
        history = get_user_data(user_id, 'current_chat', [])

        if model_config['name'] == MODELS["📷 Llama 4 Maverick"]:
            content = [{"type": "text", "text": update.message.caption or "What is in this image?"}, {"type": "image_bytes", "bytes": photo_bytes}]
            history.append({"role": "user", "content": content, "timestamp": datetime.now(timezone.utc).isoformat()})
        else:
            image_prompt = get_user_data(user_id, 'image_prompt', get_text(user_id, 'default_image_prompt'))
            b64 = base64.b64encode(photo_bytes).decode('utf-8')
            ocr_messages = [{"role": "user", "content": [{"type": "text", "text": image_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
            vision_config = {"name": MODELS["📷 Llama 4 Maverick"], "temperature": 0.5}
            completion = await get_groq_completion_with_retry(update, context, ocr_messages, vision_config, 4096)
            description = strip_think_tags(completion.choices[0].message.content)
            user_text = update.message.caption or "I've sent an image."
            history.append({"role": "user", "content": f"[Image Analysis: {description}]\n\nUser query: {user_text}", "timestamp": datetime.now(timezone.utc).isoformat()})
        await process_and_respond(update, context, history)
    except Exception as e:
        logger.error(f"Error processing photo: {e}", exc_info=True)
        await update.message.reply_text(get_text(user_id, 'photo_error'))

async def handle_unsupported_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(get_text(update.effective_user.id, 'unsupported_file_error'))
async def handle_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.delete()

def main() -> None:
    if not TELEGRAM_TOKEN:
        logger.error("Configuration error: TELEGRAM_TOKEN not set.")
        return
    load_languages()
    logger.info("Starting bot...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    cmd_handlers = {
        "start": start, "new_chat": new_chat, "api": api_command, "fallback_api": fallback_api_command,
        "models": models_command, "system_prompt": system_prompt_command, "image_prompt": image_prompt_command,
        "temperature": temperature_command, "web_search": web_search_command, "erase_me": erase_me_command,
        "max_completion_tokens": max_completion_tokens_command, "use_telegraph": use_telegraph_command,
        "language": language_command
    }
    for cmd, hnd in cmd_handlers.items():
        app.add_handler(CommandHandler(cmd, hnd))
        
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_input))
    app.add_handler(MessageHandler(filters.PHOTO, handle_user_input))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_user_input))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND & ~filters.TEXT & ~filters.PHOTO & ~filters.VOICE & ~filters.AUDIO, handle_unsupported_message))
    app.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))
    
    try:
        logger.info("Bot is polling...")
        app.run_polling()
    except Exception as e:
        logger.critical(f"Bot crashed: {e}", exc_info=True)

if __name__ == '__main__':
    main()