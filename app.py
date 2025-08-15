import os
import logging
import base64
import yaml
import re
import asyncio
import subprocess
import markdown2
from datetime import datetime, timezone
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand, BotCommandScopeChat
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.error import BadRequest
from groq import Groq, RateLimitError, APIStatusError
from ddgs import DDGS
from telegraph.aio import Telegraph
from telegraph.utils import html_to_nodes

# --- Configuration ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DATABASE_FILE = "database.yaml"
IMAGE_DIR = "image_history"
AUDIO_DIR = "audio_history"
TELEGRAM_MSG_LIMIT = 4096

DEFAULT_MODEL = { "name": "meta-llama/llama-4-maverick-17b-128e-instruct", "temperature": 1.0 }
DEFAULT_IMAGE_PROMPT = "You are an expert at analyzing images. Describe this image in detail and transcribe any and all text you see within it."
DEFAULT_MAX_TOKENS = 8192
TELEGRAPH_TABLE_HINT = (
    "\n\nIMPORTANT FORMATTING RULE: When you need to present data in a table, "
    "DO NOT use standard Markdown tables. Instead, format it as a monospaced, "
    "ASCII-style table enclosed in a markdown code block (using triple backticks ```)."
)

MODELS = { "Kimi k2": "moonshotai/kimi-k2-instruct", "📷 Llama 4 Maverick": "meta-llama/llama-4-maverick-17b-128e-instruct", "GPT-Oss 120B": "openai/gpt-oss-120b" }
MODEL_MAX_TOKENS = { "moonshotai/kimi-k2-instruct": 16384, "meta-llama/llama-4-maverick-17b-128e-instruct": 8192, "openai/gpt-oss-120b": 65536 }

# --- Logging Setup ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
os.makedirs(IMAGE_DIR, exist_ok=True); os.makedirs(AUDIO_DIR, exist_ok=True)
telegraph = Telegraph()

# --- Helper Functions ---
def escape_markdown_v2(text: str) -> str:
    escape_chars = r'_*[]()~`>#+-=|{}.!'; return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# --- DEFINITIVE FIX for IndentationError ---
def strip_think_tags(text: str) -> str:
    """Intelligently cleans the response from a reasoner model."""
    text = text.strip()
    # Case 1: <think>...</think> followed by the answer
    match = re.search(r"<think>.*?</think>(.*)", text, flags=re.DOTALL)
    if match:
        answer = match.group(1).strip()
        if answer:
            return answer
    # Case 2: Answer is wrapped inside <think>...</think>
    match = re.search(r"<think>(.*?)(</think>|$)", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: No tags found
    return text

async def send_temp_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, duration: int = 3):
    try: msg = await context.bot.send_message(chat_id=chat_id, text=text); await asyncio.sleep(duration); await msg.delete()
    except Exception as e: logger.warning(f"Could not send or delete temporary message: {e}")

async def send_response(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    user_id = update.effective_user.id
    telegraph_mode = get_user_data(user_id, 'use_telegraph', 'Never')
    use_telegraph_now = (telegraph_mode == 'Always' or (telegraph_mode == 'Long messages' and len(text) > TELEGRAM_MSG_LIMIT))
    if use_telegraph_now:
        try:
            token = get_user_data(user_id, 'telegraph_token')
            if not token: account = await telegraph.create_account(short_name=update.effective_user.first_name); token = account['access_token']; set_user_data(user_id, 'telegraph_token', token)
            user_telegraph = Telegraph(access_token=token)
            html_content = markdown2.markdown(text, extras=["fenced-code-blocks"])
            html_content = re.sub(r'<h[1-6].*?>', '<b>', html_content); html_content = re.sub(r'</h[1-6]>', '</b><br>', html_content)
            nodes = html_to_nodes(html_content)
            title = f"Response - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
            response = await user_telegraph.create_page(title=title, content=nodes)
            await update.message.reply_text(response['url'])
            return
        except Exception as e: logger.error(f"Telegraph error for user {user_id}: {e}. Falling back to splitting.")
    if len(text) > TELEGRAM_MSG_LIMIT:
        for i in range(0, len(text), TELEGRAM_MSG_LIMIT): await update.message.reply_text(text[i:i + TELEGRAM_MSG_LIMIT])
    else: await update.message.reply_text(text)

def is_api_key_valid_sync(api_key: str) -> bool:
    if not api_key or not api_key.startswith("gsk_"): return False
    try: Groq(api_key=api_key, max_retries=0).models.list(); return True
    except Exception as e: logger.warning(f"API key validation failed: {e}"); return False
def convert_audio_with_ffmpeg(input_path: str, output_path: str):
    command = ['ffmpeg', '-i', input_path, '-y', output_path]
    try: subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception as e: logger.error(f"FFmpeg failed: {e}"); raise

# --- Database Functions ---
def load_database():
    if not os.path.exists(DATABASE_FILE): return {}
    try:
        with open(DATABASE_FILE, 'r') as f: return yaml.safe_load(f) or {}
    except Exception as e: logger.error(f"Error loading database: {e}"); return {}
def save_database(data):
    try:
        with open(DATABASE_FILE, 'w') as f: yaml.dump(data, f, default_flow_style=False)
    except Exception as e: logger.error(f"Error saving database: {e}")
def get_user_data(user_id, key, default=None):
    return load_database().get(user_id, {}).get(key, default)
def set_user_data(user_id, key, value):
    db = load_database(); db.setdefault(user_id, {})[key] = value; save_database(db)

# --- Groq-related Functions ---
async def get_groq_completion_with_retry(groq_client, messages, model_config, effective_tokens, update, context):
    params = { "model": model_config['name'], "temperature": float(model_config.get('temperature', 1.0)), "max_tokens": effective_tokens, "messages": messages }
    if 'reasoning_effort' in model_config: params['reasoning_effort'] = model_config['reasoning_effort']
    while True:
        try: return await asyncio.to_thread(groq_client.chat.completions.create, **params)
        except RateLimitError as e:
            retry_after = int(e.response.headers.get("retry-after", 60)); logger.warning(f"Rate limit exceeded. Retrying after {retry_after}s.")
            await update.effective_chat.send_message(f"Too many requests. I'll automatically retry in {retry_after} seconds."); await asyncio.sleep(retry_after)
        except APIStatusError as e:
            if e.status_code == 413: await update.message.reply_text("The conversation history is too long. Please start a new chat using /new_chat.")
            else: await update.message.reply_text("An unexpected API error occurred. Please try again.")
            logger.error(f"API Status Error for user {update.effective_user.id}: {e}"); raise
        except Exception as e: logger.error(f"An unexpected error occurred calling Groq API: {e}"); raise

async def prepare_messages_for_groq(user_id, history, model_name):
    system_prompt = get_user_data(user_id, 'system_prompt', 'You are a helpful assistant.')
    telegraph_mode = get_user_data(user_id, 'use_telegraph', 'Never')
    if telegraph_mode in ['Always', 'Long messages']:
        system_prompt += TELEGRAPH_TABLE_HINT
    groq_messages = [{"role": "system", "content": f"Today is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}. {system_prompt}"}]
    is_multimodal = model_name == MODELS["📷 Llama 4 Maverick"]
    for message in history:
        msg_copy = message.copy()
        if isinstance(msg_copy.get("content"), list):
            if not is_multimodal: msg_copy["content"] = " ".join([item.get("text", "") for item in msg_copy["content"] if item.get("type") == "text"])
            else:
                new_content = []
                for item in msg_copy["content"]:
                    if item.get("type") == "image_path":
                        try:
                            with open(item["path"], "rb") as f: b64 = base64.b64encode(f.read()).decode('utf-8')
                            new_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                        except FileNotFoundError: new_content.append({"type": "text", "text": "[Image not found]"})
                    else: new_content.append(item)
                msg_copy["content"] = new_content
        msg_copy.pop("timestamp", None)
        groq_messages.append(msg_copy)
    return groq_messages

# --- Telegram Bot Command & Menu Setup ---
async def set_bot_commands_for_user(user_id: int, context: ContextTypes.DEFAULT_TYPE):
    model_config = get_user_data(user_id, 'selected_model', DEFAULT_MODEL); model_friendly_name = [name for name, id in MODELS.items() if id == model_config['name']][0]
    model_display = model_friendly_name
    if 'reasoning_effort' in model_config: model_display += f" ({model_config['reasoning_effort']})"
    user_tokens = get_user_data(user_id, 'max_completion_tokens', DEFAULT_MAX_TOKENS); model_limit = MODEL_MAX_TOKENS.get(model_config['name'], DEFAULT_MAX_TOKENS)
    effective_tokens = min(user_tokens, model_limit); tokens_display = f"current: {user_tokens} ({effective_tokens})"
    temp_display = f"current: {model_config.get('temperature', 1.0)}"
    commands = [
        BotCommand("new_chat", "start a new conversation"), BotCommand("models", f"current: {model_display}"),
        BotCommand("temperature", temp_display), BotCommand("max_completion_tokens", tokens_display),
        BotCommand("web_search", f"web search: {'ON' if get_user_data(user_id, 'web_search_enabled', False) else 'OFF'}"),
        BotCommand("use_telegraph", f"telegra.ph: {get_user_data(user_id, 'use_telegraph', 'Never')}"),
        BotCommand("system_prompt", "modify the system prompt"), BotCommand("image_prompt", "modify image description prompt"),
        BotCommand("api", "see or modify the groq api key")
    ]
    await context.bot.set_my_commands(commands, scope=BotCommandScopeChat(chat_id=user_id)); logger.info(f"Updated command menu for user {user_id}")

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    set_user_data(user_id, 'state', 'chatting')
    await set_bot_commands_for_user(user_id, context)
    if not await asyncio.to_thread(is_api_key_valid_sync, get_user_data(user_id, 'groq_api_key')):
        msg = await update.message.reply_text("Welcome! Please provide your Groq API key to begin.")
        set_user_data(user_id, 'state', 'awaiting_initial_api_key')
        set_user_data(user_id, 'last_bot_message_id', msg.message_id)
    else:
        await update.message.reply_text("Hello! I am a multimodal chatbot with memory. Use the menu button to see commands.")

async def new_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    db = load_database()
    user_data = db.setdefault(user_id, {})
    if user_data.get("current_chat"):
        user_data.setdefault('archived', {})[f"archived_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"] = user_data["current_chat"]
    user_data["current_chat"] = []
    save_database(db)
    await update.message.reply_text("Starting a new chat.")

async def generic_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, user_state: str, prompt_text: str):
    await update.message.delete()
    keyboard = [[InlineKeyboardButton("Cancel", callback_data="cancel_generic")]]
    msg = await update.effective_chat.send_message(prompt_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode='MarkdownV2')
    set_user_data(update.effective_user.id, 'state', user_state)
    set_user_data(update.effective_user.id, 'last_bot_message_id', msg.message_id)

async def api_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    key = get_user_data(update.effective_user.id, 'groq_api_key', 'Not set')
    masked = f"{key[:7]}...{key[-4:]}" if key != 'Not set' else 'Not set'
    await generic_command_handler(update, context, 'awaiting_api_key_update', f"Current Groq API:\n`{escape_markdown_v2(masked)}`\n\nInsert new key\\.")

async def system_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = get_user_data(update.effective_user.id, 'system_prompt', 'Not set')
    await generic_command_handler(update, context, 'awaiting_system_prompt', f"Current system prompt:\n`{escape_markdown_v2(prompt)}`\n\nInsert new prompt\\.")

async def image_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = get_user_data(update.effective_user.id, 'image_prompt', DEFAULT_IMAGE_PROMPT)
    await generic_command_handler(update, context, 'awaiting_image_prompt', f"Current image prompt:\n`{escape_markdown_v2(prompt)}`\n\nInsert new prompt\\.")

async def temperature_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    temp = get_user_data(update.effective_user.id, 'selected_model', DEFAULT_MODEL).get('temperature', 1.0)
    await generic_command_handler(update, context, 'awaiting_temperature', f"Current temperature: `{temp}`\\.\nEnter value from `0.0` to `1.0`\\.")

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.delete()
    keyboard = [[InlineKeyboardButton(name, callback_data=f"model_{model_id}") for name, model_id in MODELS.items()]]
    await update.effective_chat.send_message("Choose a new model.", reply_markup=InlineKeyboardMarkup(keyboard))

async def web_search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    await update.message.delete()
    new_status = not get_user_data(user_id, 'web_search_enabled', False)
    set_user_data(user_id, 'web_search_enabled', new_status)
    await send_temp_message(context, user_id, f"Web search {'ON' if new_status else 'OFF'}")
    await set_bot_commands_for_user(user_id, context)

async def max_completion_tokens_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.delete()
    keyboard = [[InlineKeyboardButton(str(val), callback_data=f"tokens_{val}") for val in [512, 1024, 2048, 4096]], [InlineKeyboardButton(str(val), callback_data=f"tokens_{val}") for val in [8192, 16384, 32768, 65536]]]
    await update.effective_chat.send_message("Select a new max completion token value.", reply_markup=InlineKeyboardMarkup(keyboard))

async def use_telegraph_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.delete()
    keyboard = [[InlineKeyboardButton(val, callback_data=f"telegraph_{val}") for val in ["Never", "Long messages", "Always"]]]
    await update.effective_chat.send_message("When should responses be sent via telegra.ph?", reply_markup=InlineKeyboardMarkup(keyboard))

async def perform_web_search(query: str):
    logger.info(f"Performing web search for: {query}");
    try:
        with DDGS() as ddgs: results = list(ddgs.text(query, max_results=3))
        if not results: return "No web search results found."
        return "\n\n--- Web Search Results ---\n" + "\n\n".join([f"Title: {r.get('title', 'N/A')}\nSnippet: {r.get('body', 'N/A')}" for r in results])
    except Exception as e: logger.error(f"Error during web search: {e}"); return None

# --- Callback and Core Logic ---
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query; await query.answer(); user_id = query.from_user.id; data = query.data
    if data.startswith("model_"):
        model_id = data.replace("model_", "")
        if model_id == MODELS["GPT-Oss 120B"]:
            keyboard = [[InlineKeyboardButton(e.title(), callback_data=f"reasoning_{e}_{model_id}") for e in ["low", "medium", "high"]]]
            await query.edit_message_text("Select reasoning effort for GPT-Oss 120B.", reply_markup=InlineKeyboardMarkup(keyboard)); return
        else: config = get_user_data(user_id, 'selected_model', DEFAULT_MODEL); set_user_data(user_id, 'selected_model', {"name": model_id, "temperature": config.get('temperature', 1.0)})
    elif data.startswith("reasoning_"):
        _, effort, model_id = data.split("_", 2); config = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
        set_user_data(user_id, 'selected_model', {"name": model_id, "temperature": config.get('temperature', 1.0), "reasoning_effort": effort})
    elif data.startswith("tokens_"): set_user_data(user_id, 'max_completion_tokens', int(data.split("_")[1]))
    elif data.startswith("telegraph_"): set_user_data(user_id, 'use_telegraph', data.split("_")[1])
    elif data == "cancel_generic": set_user_data(user_id, 'state', 'chatting')
    try: await query.message.delete()
    except: pass
    await set_bot_commands_for_user(user_id, context)

async def handle_user_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message: return
    user_id = update.message.from_user.id
    state = get_user_data(user_id, 'state')
    
    async def cleanup_and_set_state(new_state='chatting'):
        await update.message.delete()
        last_msg_id = get_user_data(user_id, 'last_bot_message_id')
        if last_msg_id:
            try: await context.bot.delete_message(chat_id=user_id, message_id=last_msg_id)
            except: pass
        set_user_data(user_id, 'state', new_state)

    state_map = {'awaiting_api_key_update': ('groq_api_key', "API updated"), 'awaiting_system_prompt': ('system_prompt', "System prompt updated!"), 'awaiting_image_prompt': ('image_prompt', "Image prompt updated!")}
    if state in state_map:
        key, msg = state_map[state]; value = update.message.text
        if key == 'groq_api_key' and not await asyncio.to_thread(is_api_key_valid_sync, value):
            await cleanup_and_set_state(); await send_temp_message(context, user_id, "API not valid."); return
        set_user_data(user_id, key, value); await cleanup_and_set_state(); await send_temp_message(context, user_id, msg); return

    elif state == 'awaiting_temperature':
        try:
            temp = float(update.message.text)
            if 0.0 <= temp <= 1.0:
                config = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
                config['temperature'] = temp; set_user_data(user_id, 'selected_model', config)
                await send_temp_message(context, user_id, f"Temperature updated to {temp}")
            else: await send_temp_message(context, user_id, "Invalid value.")
        except: await send_temp_message(context, user_id, "Invalid format.")
        await cleanup_and_set_state(); await set_bot_commands_for_user(user_id, context); return

    if not await asyncio.to_thread(is_api_key_valid_sync, get_user_data(user_id, 'groq_api_key')):
        await start(update, context); return
    
    if update.message.voice or update.message.audio: await handle_audio_message(update, context)
    elif update.message.photo: await handle_photo_message(update, context)
    elif update.message.text: await handle_text_message(update, context)

async def process_and_respond(update: Update, context: ContextTypes.DEFAULT_TYPE, history: list):
    user_id = update.effective_user.id
    api_key = get_user_data(user_id, 'groq_api_key')
    try:
        groq_client = Groq(api_key=api_key, max_retries=0)
        model_config = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
        user_tokens = get_user_data(user_id, 'max_completion_tokens', DEFAULT_MAX_TOKENS)
        model_limit = MODEL_MAX_TOKENS.get(model_config['name'], DEFAULT_MAX_TOKENS)
        effective_tokens = min(user_tokens, model_limit)
        messages = await prepare_messages_for_groq(user_id, history, model_config['name'])
        completion = await get_groq_completion_with_retry(groq_client, messages, model_config, effective_tokens, update, context)
        response_text = strip_think_tags(completion.choices[0].message.content)
        history.append({"role": "assistant", "content": response_text, "timestamp": datetime.now(timezone.utc).isoformat()})
        set_user_data(user_id, 'current_chat', history)
        await send_response(update, context, response_text)
    except (APIStatusError, RateLimitError): logger.warning(f"API call aborted for user {user_id} due to unrecoverable error.")
    except Exception as e: logger.error(f"Error in process_and_respond for user {user_id}: {e}", exc_info=True); await update.message.reply_text("An internal error occurred.")

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    prompt = update.message.text
    if get_user_data(update.effective_user.id, 'web_search_enabled', False):
        results = await perform_web_search(prompt)
        if results: prompt = f"{results}\n\nUser query: {prompt}"
        else: await send_temp_message(context, update.effective_chat.id, "❗Could not perform web search")
    history = get_user_data(update.effective_user.id, 'current_chat', []); history.append({"role": "user", "content": prompt, "timestamp": datetime.now(timezone.utc).isoformat()})
    await process_and_respond(update, context, history)

async def handle_audio_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing'); oga_path, mp3_path = None, None
    try:
        audio = update.message.voice or update.message.audio; file = await audio.get_file()
        oga_path = os.path.join(AUDIO_DIR, f"{update.effective_user.id}_{file.file_unique_id}.oga"); await file.download_to_drive(oga_path)
        mp3_path = os.path.splitext(oga_path)[0] + ".mp3"; convert_audio_with_ffmpeg(oga_path, mp3_path)
        api_key = get_user_data(update.effective_user.id, 'groq_api_key')
        with open(mp3_path, "rb") as f:
            transcription = await asyncio.to_thread(Groq(api_key=api_key, max_retries=0).audio.transcriptions.create, file=(os.path.basename(mp3_path), f.read()), model="whisper-large-v3")
        history = get_user_data(update.effective_user.id, 'current_chat', []); history.append({"role": "user", "content": f'[Transcribed from audio]: "{transcription.text}"', "timestamp": datetime.now(timezone.utc).isoformat()})
        await process_and_respond(update, context, history)
    except Exception as e: logger.error(f"Error processing audio: {e}", exc_info=True); await update.message.reply_text("Error processing audio.")
    finally:
        if oga_path and os.path.exists(oga_path): os.remove(oga_path)
        if mp3_path and os.path.exists(mp3_path): os.remove(mp3_path)

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing'); image_path = None
    try:
        user_id = update.effective_user.id; model_config = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
        photo_file = await update.message.photo[-1].get_file(); image_path = os.path.join(IMAGE_DIR, f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"); await photo_file.download_to_drive(image_path)
        history = get_user_data(user_id, 'current_chat', [])
        if model_config['name'] == MODELS["📷 Llama 4 Maverick"]:
            content = [{"type": "text", "text": update.message.caption or "What is in this image?"}, {"type": "image_path", "path": image_path}]
            history.append({"role": "user", "content": content, "timestamp": datetime.now(timezone.utc).isoformat()})
        else:
            image_prompt = get_user_data(user_id, 'image_prompt', DEFAULT_IMAGE_PROMPT)
            telegraph_mode = get_user_data(user_id, 'use_telegraph', 'Never')
            if telegraph_mode in ['Always', 'Long messages']:
                image_prompt += TELEGRAPH_TABLE_HINT
            with open(image_path, "rb") as f: b64 = base64.b64encode(f.read()).decode('utf-8')
            ocr_messages = [{"role": "user", "content": [{"type": "text", "text": image_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
            vision_config = {"name": MODELS["📷 Llama 4 Maverick"], "temperature": 0.5}; api_key = get_user_data(user_id, 'groq_api_key')
            completion = await get_groq_completion_with_retry(Groq(api_key=api_key, max_retries=0), ocr_messages, vision_config, 4096, update, context)
            description = strip_think_tags(completion.choices[0].message.content)
            user_text = update.message.caption or "I've sent an image."
            history.append({"role": "user", "content": f"[Image Analysis: {description}]\n\nUser query: {user_text}", "timestamp": datetime.now(timezone.utc).isoformat()})
        await process_and_respond(update, context, history)
    except Exception as e: logger.error(f"Error processing photo: {e}", exc_info=True); await update.message.reply_text("Error with the image.")
    finally:
        if image_path and os.path.exists(image_path): os.remove(image_path)

async def handle_unsupported_message(update: Update, context: ContextTypes.DEFAULT_TYPE): await update.message.reply_text("Sorry, I can only process text, photo, and audio messages.")
async def handle_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE): await update.message.delete()

def main() -> None:
    if not TELEGRAM_TOKEN: logger.error("Configuration error: TELEGRAM_TOKEN not set."); return
    logger.info("Starting bot...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    cmd_handlers = {"start": start, "new_chat": new_chat, "api": api_command, "models": models_command, "system_prompt": system_prompt_command, "image_prompt": image_prompt_command, "temperature": temperature_command, "web_search": web_search_command, "max_completion_tokens": max_completion_tokens_command, "use_telegraph": use_telegraph_command}
    for cmd, hnd in cmd_handlers.items(): app.add_handler(CommandHandler(cmd, hnd))
    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_input))
    app.add_handler(MessageHandler(filters.PHOTO, handle_user_input))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_user_input))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND & ~filters.TEXT & ~filters.PHOTO & ~filters.VOICE & ~filters.AUDIO, handle_unsupported_message))
    app.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))
    try: logger.info("Bot is polling..."); app.run_polling()
    except Exception as e: logger.error(f"Bot crashed: {e}", exc_info=True)

if __name__ == '__main__':
    main()
