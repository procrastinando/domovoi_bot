import asyncio
import logging
import base64
from datetime import datetime, timezone
from io import BytesIO
import subprocess

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand, BotCommandScopeChat
from telegram.ext import ContextTypes
from groq import Groq, RateLimitError, APIStatusError

from bot.database import get_user_data, set_user_data, load_database, save_database
from bot.groq_client import get_groq_completion_with_retry, prepare_messages_for_groq, transcribe_audio_with_retry
from bot.localization import get_text, SUPPORTED_LANGUAGES
from bot.utils import send_temp_message, send_response, escape_markdown_v2, is_api_key_valid_sync, perform_web_search, strip_think_tags
from config import DEFAULT_MODEL, DEFAULT_MAX_TOKENS, MODEL_MAX_TOKENS, MODELS, VISION_MODELS

logger = logging.getLogger(__name__)

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
    prompt = get_text(user_id, 'api_key_prompt', masked_key=masked)
    await generic_command_handler(update, context, 'awaiting_api_key', prompt)

async def fallback_api_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    key = get_user_data(user_id, 'fallback_api_key', 'Not set')
    masked = f"{key[:7]}...{key[-4:]}" if '...' not in key else 'Not set'
    prompt = get_text(user_id, 'fallback_api_key_prompt', masked_key=masked)
    await generic_command_handler(update, context, 'awaiting_fallback_api_key', prompt)

async def system_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    prompt = get_user_data(user_id, 'system_prompt', get_text(user_id, 'default_system_prompt'))
    text = get_text(user_id, 'system_prompt_prompt', prompt=prompt)
    await generic_command_handler(update, context, 'awaiting_system_prompt', text)

async def image_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    prompt = get_user_data(user_id, 'image_prompt', get_text(user_id, 'default_image_prompt'))
    text = get_text(user_id, 'image_prompt_prompt', prompt=prompt)
    await generic_command_handler(update, context, 'awaiting_image_prompt', text)

async def temperature_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    temp = get_user_data(user_id, 'selected_model', DEFAULT_MODEL).get('temperature', 1.0)
    text = get_text(user_id, 'temperature_prompt', temp=temp)
    await generic_command_handler(update, context, 'awaiting_temperature', text)

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
        try:
            await query.message.delete()
        except:
            pass

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
            if 0.0 <= temp <= 2.0:
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
        logger.info(f"Running ffmpeg command for user {user_id}: {' '.join(command)}")
        process = await asyncio.create_subprocess_exec(
            *command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        converted_audio_bytes, stderr = await process.communicate(input=audio_bytes_io.read())
        if process.returncode != 0:
            logger.error(f"FFmpeg failed for user {user_id}: {stderr.decode()}")
            raise IOError(f"FFmpeg failed: {stderr.decode()}")
        logger.info(f"FFmpeg conversion successful for user {user_id}")

        transcription = await transcribe_audio_with_retry(update, context, converted_audio_bytes)
        logger.info(f"Transcription successful for user {user_id}")
        history = get_user_data(user_id, 'current_chat', [])
        history.append({"role": "user", "content": f'[Transcribed from audio]: "{transcription.text}"' , "timestamp": datetime.now(timezone.utc).isoformat()})
        await process_and_respond(update, context, history)
    except Exception as e:
        logger.error(f"Error processing audio for user {user_id}: {e}", exc_info=True)
        await update.message.reply_text(get_text(user_id, 'audio_error'))

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    try:
        model_config = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
        photo_file = await update.message.photo[-1].get_file()

        photo_bytes = bytes(await photo_file.download_as_bytearray())
        history = get_user_data(user_id, 'current_chat', [])

        if model_config['name'] not in VISION_MODELS:
            # Get description from vision model
            image_prompt = get_user_data(user_id, 'image_prompt', get_text(user_id, 'default_image_prompt'))

            b64_image = base64.b64encode(photo_bytes).decode('utf-8')
            vision_messages = [
                {"role": "system", "content": image_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]}
            ]

            vision_model_config = {"name": VISION_MODELS[0], "temperature": 0.0}

            completion = await get_groq_completion_with_retry(update, context, vision_messages, vision_model_config, 4096)
            description = completion.choices[0].message.content

            # Add description to history and continue with original model
            prompt = f"[Image Description]: {description}\n\n{update.message.caption or ''}"
            history.append({"role": "user", "content": prompt, "timestamp": datetime.now(timezone.utc).isoformat()})
        else:
            content = [{"type": "text", "text": update.message.caption or "What is in this image?"}, {"type": "image_bytes", "bytes": photo_bytes}]
            history.append({"role": "user", "content": content, "timestamp": datetime.now(timezone.utc).isoformat()})

        await process_and_respond(update, context, history)
    except Exception as e:
        logger.error(f"Error processing photo: {e}", exc_info=True)
        await update.message.reply_text(get_text(user_id, 'photo_error'))

async def handle_unsupported_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(get_text(update.effective_user.id, 'unsupported_file_error'))

async def handle_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.delete()