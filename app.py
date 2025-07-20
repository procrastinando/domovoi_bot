import os
import logging
import base64
import yaml
import re
import asyncio
import subprocess
from io import BytesIO
from datetime import datetime, timezone
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand, BotCommandScopeChat
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.error import BadRequest
from groq import Groq, RateLimitError, APIStatusError
from ddgs import DDGS

# --- Configuration ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DATABASE_FILE = "database.yaml"
IMAGE_DIR = "image_history"
AUDIO_DIR = "audio_history"
DEFAULT_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

MODELS = {
    "🧠 Kimi k2": "moonshotai/kimi-k2-instruct",
    "📷 Llama 4 Maverick": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "Deepseek": "deepseek-r1-distill-llama-70b"
}

# --- Logging Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# --- Helper Functions ---
def escape_markdown_v2(text: str) -> str:
    """Escapes characters for Telegram's MarkdownV2 parse mode."""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

def strip_think_tags(text: str) -> str:
    """
    Intelligently cleans the response from a reasoner model.
    - If the response is in the format <think>...</think>answer, it returns the answer.
    - If the response is in the format <think>answer</think>, it extracts and returns the answer.
    """
    text = text.strip()
    
    # Case 1: The model provides thoughts AND a final answer separately.
    # The pattern finds a complete <think>...</think> block and captures what's AFTER it.
    match = re.search(r"<think>.*?</think>(.*)", text, flags=re.DOTALL)
    if match:
        # If there's content after the block, that's our answer.
        answer = match.group(1).strip()
        if answer:
            return answer
            
    # Case 2: The model wraps its entire response in <think> tags.
    # The pattern captures what's INSIDE the <think> tag. It handles a missing </think>.
    match = re.search(r"<think>(.*?)(</think>|$)", text, flags=re.DOTALL)
    if match:
        # The answer is the content inside the block.
        return match.group(1).strip()
        
    # Fallback: If no <think> tags are found, return the original text as is.
    return text

async def send_temp_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, duration: int = 3):
    """Sends a temporary message that is deleted after a few seconds."""
    try:
        msg = await context.bot.send_message(chat_id=chat_id, text=text)
        await asyncio.sleep(duration)
        await msg.delete()
    except Exception as e:
        logger.warning(f"Could not send or delete temporary message: {e}")

def is_api_key_valid_sync(api_key: str) -> bool:
    """Synchronous check if a Groq API key is valid."""
    if not api_key or not api_key.startswith("gsk_"):
        return False
    try:
        test_client = Groq(api_key=api_key)
        test_client.models.list()
        return True
    except (RateLimitError, APIStatusError) as e:
        logger.warning(f"API key validation failed for key ending in ...{api_key[-4:]}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during API key validation: {e}")
        return False

def convert_audio_with_ffmpeg(input_path: str, output_path: str):
    """Converts audio file using ffmpeg in a blocking manner."""
    logger.info(f"Starting ffmpeg conversion from {input_path} to {output_path}")
    command = ['ffmpeg', '-i', input_path, '-y', output_path]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"FFmpeg conversion successful for {output_path}")
    except FileNotFoundError:
        logger.error("ffmpeg command not found. Please ensure ffmpeg is installed and in your system's PATH.")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed with exit code {e.returncode}. Stderr: {e.stderr}")
        raise

# --- Database Functions ---
def load_database():
    if not os.path.exists(DATABASE_FILE): return {}
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
    db = load_database()
    return db.get(user_id, {}).get(key, default)

def set_user_data(user_id, key, value):
    db = load_database()
    db.setdefault(user_id, {})[key] = value
    save_database(db)

# --- Groq-related Functions ---
async def get_groq_completion_with_retry(groq_client, messages, model, update, context):
    """Calls Groq API and handles rate limit errors with automatic retry."""
    while True:
        try:
            chat_completion = await asyncio.to_thread(
                groq_client.chat.completions.create, messages=messages, model=model
            )
            return chat_completion
        except RateLimitError as e:
            retry_after_str = e.response.headers.get("retry-after")
            retry_after_seconds = int(retry_after_str) if retry_after_str else 60
            logger.warning(f"Rate limit exceeded. Retrying after {retry_after_seconds} seconds.")
            await update.effective_chat.send_message(
                f"Too many requests. I'll automatically retry in {retry_after_seconds} seconds."
            )
            await asyncio.sleep(retry_after_seconds)
            # Loop will continue and retry the request
        except Exception as e:
            logger.error(f"An unexpected error occurred calling Groq API: {e}")
            raise # Re-raise other exceptions to be handled by the calling function

async def prepare_messages_for_groq(user_id, history, model):
    # (This function remains unchanged)
    groq_messages = []
    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    user_prompt = get_user_data(user_id, 'system_prompt', "You are a helpful assistant.")
    final_system_prompt = f"Today is {now_utc}. {user_prompt}"
    if final_system_prompt and final_system_prompt != "Not set":
        groq_messages.append({"role": "system", "content": final_system_prompt})

    is_multimodal = model == MODELS["Llama 4 Maverick 📷"]
    for message in history:
        msg_copy = message.copy()
        content = msg_copy.get("content")
        if isinstance(content, list):
            if not is_multimodal:
                msg_copy["content"] = " ".join([item.get("text","") for item in content if item.get("type") == "text"])
            else:
                new_content_list = []
                for item in content:
                    if item.get("type") == "image_path":
                        try:
                            with open(item["path"], "rb") as f:
                                base64_image = base64.b64encode(f.read()).decode('utf-8')
                            new_content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
                        except FileNotFoundError:
                            logger.error(f"Image file not found: {item['path']}")
                            new_content_list.append({"type": "text", "text": f"[Image not found at {item['path']}]"})
                    else:
                        new_content_list.append(item)
                msg_copy["content"] = new_content_list
        msg_copy.pop("timestamp", None)
        groq_messages.append(msg_copy)
    return groq_messages

# --- Telegram Bot Command & Menu Setup ---

async def set_bot_commands_for_user(user_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Sets the bot's command menu for a specific user with dynamic descriptions."""
    web_search_enabled = get_user_data(user_id, 'web_search_enabled', False)
    web_search_status = "ON" if web_search_enabled else "OFF"
    
    commands = [
        BotCommand("new_chat", "start a new conversation"),
        BotCommand("models", "list of available models"),
        BotCommand("web_search", f"web search, currently {web_search_status}"),
        BotCommand("system_prompt", "modify the system prompt"),
        BotCommand("api", "see or modify the groq api key")
    ]
    await context.bot.set_my_commands(commands, scope=BotCommandScopeChat(chat_id=user_id))
    logger.info(f"Updated command menu for user {user_id}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    set_user_data(user_id, 'state', 'chatting')
    await set_bot_commands_for_user(user_id, context) # Set commands on start
    api_key = get_user_data(user_id, 'groq_api_key')
    is_valid = await asyncio.to_thread(is_api_key_valid_sync, api_key)
    if not is_valid:
        sent_message = await update.message.reply_text("Welcome! Please provide your Groq API key to begin.")
        set_user_data(user_id, 'state', 'awaiting_initial_api_key')
        set_user_data(user_id, 'last_bot_message_id', sent_message.message_id)
    else:
        message = "Hello! I am a multimodal chatbot with memory, powered by Groq. You can now send voice messages for transcription. Use the menu button to see available commands."
        await update.message.reply_text(message)

async def new_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # (This function remains unchanged)
    user_id = update.message.from_user.id
    db = load_database()
    user_data = db.setdefault(user_id, {})
    if user_data.get("current_chat"):
        now = datetime.now(timezone.utc)
        archive_key = f"archived_{now.strftime('%Y%m%d%H%M%S')}"
        user_data[archive_key] = user_data["current_chat"]
    user_data["current_chat"] = []
    save_database(db)
    await update.message.reply_text("Starting a new chat. Your previous conversation has been archived.")

async def api_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # (This function remains unchanged)
    user_id = update.message.from_user.id
    await update.message.delete()
    api_key = get_user_data(user_id, 'groq_api_key', 'Not set')
    masked_key = f"{api_key[:7]}...{api_key[-4:]}" if api_key != 'Not set' else 'Not set'
    keyboard = [[InlineKeyboardButton("Cancel", callback_data="cancel_api_update")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    sent_message = await update.effective_chat.send_message(f"Current Groq API:\n\n`{escape_markdown_v2(masked_key)}`\n\nInsert a new API key\\.", reply_markup=reply_markup, parse_mode='MarkdownV2')
    set_user_data(user_id, 'state', 'awaiting_api_key_update')
    set_user_data(user_id, 'last_bot_message_id', sent_message.message_id)

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # (This function remains unchanged)
    user_id = update.message.from_user.id
    current_model = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
    current_model_name = [name for name, id in MODELS.items() if id == current_model][0]
    await update.message.delete()
    keyboard = [[InlineKeyboardButton(name, callback_data=f"model_{model_id}") for name, model_id in MODELS.items()]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    sent_message = await update.effective_chat.send_message(f"The current model is {current_model_name}, choose a new model.", reply_markup=reply_markup)
    set_user_data(user_id, 'last_bot_message_id', sent_message.message_id)

async def system_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # (This function remains unchanged)
    user_id = update.message.from_user.id
    await update.message.delete()
    current_prompt = get_user_data(user_id, 'system_prompt', "Not set")
    keyboard = [[InlineKeyboardButton("Cancel", callback_data="cancel_system_prompt")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    message_text = f"Current system prompt:\n\n`{escape_markdown_v2(current_prompt)}`\n\nInsert a new system prompt\\."
    sent_message = await update.effective_chat.send_message(message_text, reply_markup=reply_markup, parse_mode='MarkdownV2')
    set_user_data(user_id, 'state', 'awaiting_system_prompt')
    set_user_data(user_id, 'last_bot_message_id', sent_message.message_id)

async def web_search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.delete()
    user_id = update.message.from_user.id
    new_status = not get_user_data(user_id, 'web_search_enabled', False)
    set_user_data(user_id, 'web_search_enabled', new_status)
    status_text = "ON" if new_status else "OFF"
    await send_temp_message(context, update.effective_chat.id, f"Web search {status_text}")
    await set_bot_commands_for_user(user_id, context) # Update menu with new status

async def perform_web_search(query: str) -> str:
    # (This function remains unchanged)
    logger.info(f"Performing web search for: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results: return "No web search results found."
        formatted_results = "\n\n--- Web Search Results ---\n"
        for result in results:
            formatted_results += f"Title: {result.get('title', 'N/A')}\nLink: {result.get('href', 'N/A')}\nSnippet: {result.get('body', 'N/A')}\n\n"
        formatted_results += "--------------------------\n"
        return formatted_results
    except Exception as e:
        logger.error(f"Error during web search: {e}")
        return "An error occurred during web search."

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # (This function remains unchanged)
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data
    if data.startswith("model_"):
        model_id = data.replace("model_", "")
        model_name = [name for name, id in MODELS.items() if id == model_id][0]
        set_user_data(user_id, 'selected_model', model_id)
        await query.answer(text=f"Model changed to {model_name}", show_alert=False)
    elif data == "cancel_system_prompt":
        set_user_data(user_id, 'state', 'chatting')
        await query.answer(text="Operation cancelled", show_alert=False)
    elif data == "cancel_api_update":
        set_user_data(user_id, 'state', 'chatting')
        await query.answer(text="No changes in the API", show_alert=False)
    try:
        await query.message.delete()
    except (BadRequest, Exception): pass

# --- Core Logic Handlers ---

async def handle_user_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    user_id = update.message.from_user.id
    state = get_user_data(user_id, 'state')
    last_bot_message_id = get_user_data(user_id, 'last_bot_message_id')
    if state in ['awaiting_initial_api_key', 'awaiting_api_key_update']:
        new_api_key = update.message.text
        await update.message.delete()
        is_valid = await asyncio.to_thread(is_api_key_valid_sync, new_api_key)
        if is_valid:
            set_user_data(user_id, 'groq_api_key', new_api_key)
            set_user_data(user_id, 'state', 'chatting')
            if last_bot_message_id:
                try: await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=last_bot_message_id)
                except BadRequest: pass
            if state == 'awaiting_initial_api_key':
                await send_temp_message(context, user_id, "API key is valid. You can now start chatting!")
            else:
                await send_temp_message(context, user_id, "API updated")
        else:
            await send_temp_message(context, user_id, "API not valid")
        return
    elif state == 'awaiting_system_prompt':
        set_user_data(user_id, 'system_prompt', update.message.text)
        set_user_data(user_id, 'state', 'chatting')
        await update.message.delete()
        if last_bot_message_id:
            try: await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=last_bot_message_id)
            except BadRequest: pass
        await send_temp_message(context, user_id, "System prompt updated!")
        return
    api_key = get_user_data(user_id, 'groq_api_key')
    if not await asyncio.to_thread(is_api_key_valid_sync, api_key):
        await start(update, context)
        return
    if update.message.voice or update.message.audio: await handle_audio_message(update, context, api_key)
    elif update.message.photo: await handle_photo_message(update, context, api_key)
    elif update.message.text: await handle_text_message(update, context, api_key)

async def handle_audio_message(update: Update, context: ContextTypes.DEFAULT_TYPE, api_key: str) -> None:
    user_id = update.message.from_user.id
    original_audio_path, converted_audio_path = None, None
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    try:
        audio_obj = update.message.voice or update.message.audio
        file = await audio_obj.get_file()
        original_audio_path = os.path.join(AUDIO_DIR, f"{user_id}_{file.file_unique_id}.oga")
        await file.download_to_drive(original_audio_path)
        
        converted_audio_path = os.path.splitext(original_audio_path)[0] + ".mp3"
        await asyncio.to_thread(convert_audio_with_ffmpeg, original_audio_path, converted_audio_path)
        
        groq_client = Groq(api_key=api_key)
        with open(converted_audio_path, "rb") as audio_file:
            transcription = await asyncio.to_thread(
                groq_client.audio.transcriptions.create,
                file=(os.path.basename(converted_audio_path), audio_file.read()),
                model="whisper-large-v3-turbo",
            )
        transcribed_text = transcription.text
        logger.info(f"Transcription for user {user_id}: '{transcribed_text}'")
        
        history = get_user_data(user_id, 'current_chat', [])
        history.append({"role": "user", "content": f'[Transcribed from audio]: "{transcribed_text}"', "timestamp": datetime.now(timezone.utc).isoformat()})
        model = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
        messages_for_groq = await prepare_messages_for_groq(user_id, history, model)
        
        chat_completion = await get_groq_completion_with_retry(groq_client, messages_for_groq, model, update, context)
        response_text = chat_completion.choices[0].message.content
        response_text = strip_think_tags(response_text)
        history.append({"role": "assistant", "content": response_text, "timestamp": datetime.now(timezone.utc).isoformat()})
        set_user_data(user_id, 'current_chat', history)
        await update.message.reply_text(response_text)

    except Exception as e:
        logger.error(f"Error processing audio message: {e}", exc_info=True)
        await update.message.reply_text("An error occurred during audio processing.")
    finally:
        for path in [original_audio_path, converted_audio_path]:
            if path and os.path.exists(path):
                os.remove(path)

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE, api_key: str) -> None:
    user_id = update.message.from_user.id
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        prompt_message = update.message.text
        if get_user_data(user_id, 'web_search_enabled', False):
            search_results = await perform_web_search(prompt_message)
            prompt_message = f"{search_results}\n\nBased on the above web search results, please answer: {prompt_message}"
        
        groq_client = Groq(api_key=api_key)
        history = get_user_data(user_id, 'current_chat', [])
        history.append({"role": "user", "content": prompt_message, "timestamp": datetime.now(timezone.utc).isoformat()})
        model = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
        messages_for_groq = await prepare_messages_for_groq(user_id, history, model)
        
        chat_completion = await get_groq_completion_with_retry(groq_client, messages_for_groq, model, update, context)
        response_text = chat_completion.choices[0].message.content
        response_text = strip_think_tags(response_text)
        history.append({"role": "assistant", "content": response_text, "timestamp": datetime.now(timezone.utc).isoformat()})
        set_user_data(user_id, 'current_chat', history)
        await update.message.reply_text(response_text)
    except Exception as e:
        logger.error(f"Error processing text message: {e}", exc_info=True)
        await update.message.reply_text("An error occurred.")

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE, api_key: str) -> None:
    user_id = update.message.from_user.id
    model = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
    if model != MODELS["Llama 4 Maverick 📷"]:
        await update.message.reply_text(f"The current model ('{[k for k,v in MODELS.items() if v == model][0]}') does not support images. Use /models to switch.")
        return
    
    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        groq_client = Groq(api_key=api_key)
        photo_file = await update.message.photo[-1].get_file()
        image_path = os.path.join(IMAGE_DIR, f"{user_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.jpg")
        await photo_file.download_to_drive(image_path)
        
        history = get_user_data(user_id, 'current_chat', [])
        history_content = [{"type": "text", "text": update.message.caption or "What is in this image?"}, {"type": "image_path", "path": image_path}]
        history.append({"role": "user", "content": history_content, "timestamp": datetime.now(timezone.utc).isoformat()})
        messages_for_groq = await prepare_messages_for_groq(user_id, history, model)
        
        chat_completion = await get_groq_completion_with_retry(groq_client, messages_for_groq, model, update, context)
        response_text = chat_completion.choices[0].message.content
        response_text = strip_think_tags(response_text)
        history.append({"role": "assistant", "content": response_text, "timestamp": datetime.now(timezone.utc).isoformat()})
        set_user_data(user_id, 'current_chat', history)
        await update.message.reply_text(response_text)
    except Exception as e:
        logger.error(f"Error processing photo: {e}", exc_info=True)
        await update.message.reply_text("An error occurred processing the image.")

async def handle_unsupported_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles any message type not explicitly supported."""
    await update.message.reply_text("Sorry, I can only process text, photo, and audio messages at the moment.")

async def handle_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Deletes any command that is not recognized."""
    await update.message.delete()
    logger.info(f"Deleted unknown command from user {update.message.from_user.id}: {update.message.text}")

def main() -> None:
    if not TELEGRAM_TOKEN:
        logger.error("Configuration error: TELEGRAM_TOKEN not set.")
        return
    logger.info("Starting bot...")
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # --- Register Handlers ---
    # Specific commands first
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("new_chat", new_chat))
    application.add_handler(CommandHandler("api", api_command))
    application.add_handler(CommandHandler("models", models_command))
    application.add_handler(CommandHandler("system_prompt", system_prompt_command))
    application.add_handler(CommandHandler("web_search", web_search_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Handlers for specific message types
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_input))
    application.add_handler(MessageHandler(filters.PHOTO, handle_user_input))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_user_input))

    # Handler for any other message type (unsupported)
    UNSUPPORTED_FILTERS = filters.ALL & ~filters.COMMAND & ~filters.TEXT & ~filters.PHOTO & ~filters.VOICE & ~filters.AUDIO
    application.add_handler(MessageHandler(UNSUPPORTED_FILTERS, handle_unsupported_message))

    # Handler for any unknown command (must be last command-related handler)
    application.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))

    try:
        logger.info("Bot is polling...")
        application.run_polling()
    except Exception as e:
        logger.error(f"Bot crashed with error: {e}", exc_info=True)

if __name__ == '__main__':
    main()