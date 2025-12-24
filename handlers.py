import base64
from io import BytesIO
import asyncio
from groq import Groq

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand, BotCommandScopeChat
from telegram.ext import ContextTypes

from database import get_user_data, set_user_data, delete_user_data
from groq_client import execute_groq_request, transcribe_audio, prepare_messages
from search_client import perform_tavily_search, perform_compound_search, generate_search_params
from localization import get_text
from utils import (
    send_response, send_ephemeral, delete_trigger_message, strip_think_tags, 
    mask_api_key, log_activity, setup_logger, get_tavily_usage_stats, escape_html,
    get_random_api_key
)
from config import MODELS, DEFAULT_MODEL_KEY, SUPPORTED_LANGUAGES, DEFAULT_MAX_TOKENS

logger = setup_logger("handlers")

# --- MENU MANAGER ---
async def update_bot_menu(user_id: int, context: ContextTypes.DEFAULT_TYPE):
    # ... (Same as previous, omitted for brevity) ...
    model_key = get_user_data(user_id, 'selected_model_key', DEFAULT_MODEL_KEY)
    model_name = MODELS[model_key]['name']
    ws_mode = get_user_data(user_id, 'web_search_mode', 'off').upper()
    temp = get_user_data(user_id, 'temperature', 0.6)
    tokens = get_user_data(user_id, 'max_completion_tokens', DEFAULT_MAX_TOKENS)
    telegraph_mode = get_user_data(user_id, 'use_telegraph', 'Never')
    
    commands = [
        BotCommand("new_chat", get_text(user_id, 'menu_new_chat')),
        BotCommand("delete_last", get_text(user_id, 'menu_delete_last')),
        BotCommand("models", f"{get_text(user_id, 'menu_models')}: {model_name}"),
        BotCommand("web_search", f"{get_text(user_id, 'menu_web_search')}: {ws_mode}"),
        BotCommand("prompt", get_text(user_id, 'menu_prompt')),
        BotCommand("use_telegraph", f"{get_text(user_id, 'menu_use_telegraph', mode=telegraph_mode)}"),
        BotCommand("temperature", f"{get_text(user_id, 'menu_temperature', temp=temp)}"),
        BotCommand("max_completion_tokens", f"{get_text(user_id, 'menu_max_tokens', tokens=tokens)}"),
        BotCommand("language", get_text(user_id, 'menu_language')),
        BotCommand("api", get_text(user_id, 'menu_api')),
        BotCommand("erase_me", get_text(user_id, 'menu_erase_me')),
    ]
    try:
        await context.bot.set_my_commands(commands, scope=BotCommandScopeChat(chat_id=user_id))
    except Exception:
        pass

# --- COMMAND HANDLERS (Standard commands...) ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not get_user_data(user_id, 'language'):
        lang = 'es' if update.effective_user.language_code and 'es' in update.effective_user.language_code else 'en'
        set_user_data(user_id, 'language', lang)
    if not get_user_data(user_id, 'system_prompt'):
        set_user_data(user_id, 'system_prompt', get_text(user_id, 'default_system_prompt'))
        set_user_data(user_id, 'image_prompt', get_text(user_id, 'default_image_prompt'))
        set_user_data(user_id, 'websearch_prompt', get_text(user_id, 'default_websearch_prompt'))
    await update_bot_menu(user_id, context)
    if not get_user_data(user_id, 'groq_api_key'):
        await update.message.reply_text(get_text(user_id, 'welcome_prompt'), parse_mode='HTML')
        set_user_data(user_id, 'state', 'awaiting_groq_api_key')
    else:
        await update.message.reply_text(get_text(user_id, 'hello_message'))

async def new_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    set_user_data(user_id, 'current_chat', [])
    msg_text = get_text(user_id, 'new_chat_started')
    
    main_key = get_user_data(user_id, 'tavily_api_key')
    fallback_key = get_user_data(user_id, 'tavily_fallback_api_key')
    stats_lines = []
    if main_key:
        usage = await get_tavily_usage_stats(main_key)
        stats_lines.append(get_text(user_id, 'tavily_main_usage', usage=usage))
    if fallback_key:
        usage = await get_tavily_usage_stats(fallback_key)
        stats_lines.append(get_text(user_id, 'tavily_fallback_usage', usage=usage))
    if stats_lines:
        msg_text += "\n\n" + "\n".join(stats_lines)
    await update.message.reply_text(msg_text)

async def delete_last_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await delete_trigger_message(update)
    history = get_user_data(user_id, 'current_chat', [])
    if len(history) >= 2:
        bot_msg = history.pop() 
        user_msg = history.pop()
        set_user_data(user_id, 'current_chat', history)
        try:
            if bot_msg.get('message_id'):
                await context.bot.delete_message(chat_id=user_id, message_id=bot_msg['message_id'])
            if user_msg.get('message_id'):
                await context.bot.delete_message(chat_id=user_id, message_id=user_msg['message_id'])
        except Exception:
            pass 
    else:
        await send_ephemeral(context, user_id, get_text(user_id, 'msg_delete_fail'), 2)

async def generic_menu_command_handler(update: Update, context, menu_type):
    user_id = update.effective_user.id
    await delete_trigger_message(update)
    # ... (Same menu generation logic as previous, no logic change needed here) ...
    markup = None
    text = ""
    # (Pasting truncated menu logic for brevity - assume standard implementation from previous steps)
    if menu_type == 'models':
        buttons = []
        for key, info in MODELS.items():
            buttons.append(InlineKeyboardButton(info['name'], callback_data=f"set_model_{key}"))
        rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
        rows.append([InlineKeyboardButton(get_text(user_id, 'button_cancel'), callback_data="cancel")])
        text = get_text(user_id, 'models_prompt')
        markup = InlineKeyboardMarkup(rows)
    elif menu_type == 'web_search':
        buttons = [
            InlineKeyboardButton(get_text(user_id, 'ws_mode_off'), callback_data="ws_mode_off"),
            InlineKeyboardButton(get_text(user_id, 'ws_mode_compound'), callback_data="ws_mode_compound"),
            InlineKeyboardButton(get_text(user_id, 'ws_mode_tavily'), callback_data="ws_mode_tavily")
        ]
        rows = [buttons[i:i + 3] for i in range(0, len(buttons), 3)]
        rows.append([InlineKeyboardButton(get_text(user_id, 'button_cancel'), callback_data="cancel")])
        text = f"{get_text(user_id, 'menu_web_search')}: {get_user_data(user_id, 'web_search_mode', 'off').upper()}"
        markup = InlineKeyboardMarkup(rows)
    elif menu_type == 'prompt':
        buttons = [
            InlineKeyboardButton("🧠 System", callback_data="set_prompt_system"),
            InlineKeyboardButton("🖼️ Image", callback_data="set_prompt_image"),
            InlineKeyboardButton("🔍 Web Search", callback_data="set_prompt_web")
        ]
        rows = [buttons[i:i + 3] for i in range(0, len(buttons), 3)]
        rows.append([InlineKeyboardButton(get_text(user_id, 'button_cancel'), callback_data="cancel")])
        text = get_text(user_id, 'menu_prompt_title')
        markup = InlineKeyboardMarkup(rows)
    elif menu_type == 'api':
        buttons = [
            InlineKeyboardButton("🔑 Groq Main", callback_data="set_api_groq_main"),
            InlineKeyboardButton("🔑 Groq Fallback", callback_data="set_api_groq_fallback"),
            InlineKeyboardButton("🌐 Tavily Main", callback_data="set_api_tavily_main"),
            InlineKeyboardButton("🌐 Tavily Fallback", callback_data="set_api_tavily_fallback")
        ]
        rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
        rows.append([InlineKeyboardButton(get_text(user_id, 'button_cancel'), callback_data="cancel")])
        text = get_text(user_id, 'menu_api')
        markup = InlineKeyboardMarkup(rows)
    elif menu_type == 'language':
        buttons = [InlineKeyboardButton(name, callback_data=f"set_lang_{code}") for code, name in SUPPORTED_LANGUAGES.items()]
        rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
        rows.append([InlineKeyboardButton(get_text(user_id, 'button_cancel'), callback_data="cancel")])
        text = "Select Language:"
        markup = InlineKeyboardMarkup(rows)
    elif menu_type == 'temperature':
        buttons = [[InlineKeyboardButton(get_text(user_id, 'button_cancel'), callback_data="cancel")]]
        text = get_text(user_id, 'temperature_prompt', temp=get_user_data(user_id, 'temperature', 0.6))
        markup = InlineKeyboardMarkup(buttons)
        set_user_data(user_id, 'state', 'awaiting_temperature')
    elif menu_type == 'tokens':
        buttons = [
            InlineKeyboardButton("2048", callback_data="tok_2048"), InlineKeyboardButton("4096", callback_data="tok_4096"),
            InlineKeyboardButton("8192", callback_data="tok_8192"), InlineKeyboardButton("16384", callback_data="tok_16384"),
            InlineKeyboardButton("32768", callback_data="tok_32768"), InlineKeyboardButton("65536", callback_data="tok_65536")
        ]
        rows = [buttons[i:i + 3] for i in range(0, len(buttons), 3)]
        rows.append([InlineKeyboardButton(get_text(user_id, 'button_cancel'), callback_data="cancel")])
        text = get_text(user_id, 'max_tokens_prompt')
        markup = InlineKeyboardMarkup(rows)
    elif menu_type == 'telegraph':
        buttons = [
            InlineKeyboardButton(get_text(user_id, 'tg_never'), callback_data="tg_Never"),
            InlineKeyboardButton(get_text(user_id, 'tg_long_messages'), callback_data="tg_Long messages"),
            InlineKeyboardButton(get_text(user_id, 'tg_always'), callback_data="tg_Always")
        ]
        rows = [buttons[i:i + 3] for i in range(0, len(buttons), 3)]
        rows.append([InlineKeyboardButton(get_text(user_id, 'button_cancel'), callback_data="cancel")])
        text = get_text(user_id, 'use_telegraph_prompt')
        markup = InlineKeyboardMarkup(rows)
    elif menu_type == 'erase':
        buttons = [[InlineKeyboardButton(get_text(user_id, 'button_yes_erase'), callback_data="erase_confirm"),
                    InlineKeyboardButton(get_text(user_id, 'button_cancel'), callback_data="cancel")]]
        text = get_text(user_id, 'erase_me_prompt')
        markup = InlineKeyboardMarkup(buttons)

    msg = await update.message.reply_text(text, reply_markup=markup, parse_mode='HTML')
    set_user_data(user_id, 'config_msg_id', msg.message_id)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data
    await query.answer()

    if data == 'cancel':
        set_user_data(user_id, 'state', 'chatting')
        await query.message.delete()
        return

    async def notify_and_close(text_key, **kwargs):
        await query.message.delete()
        text = get_text(user_id, text_key, **kwargs)
        await send_ephemeral(context, user_id, text, 3)
        await update_bot_menu(user_id, context)

    # ... (Web search menu handling same as previous) ...
    if data == 'ws_mode_tavily':
        buttons = [
            InlineKeyboardButton("None+Basic", callback_data="tavconf_none_basic"),
            InlineKeyboardButton("Basic+Basic", callback_data="tavconf_basic_basic"),
            InlineKeyboardButton("Adv+Basic", callback_data="tavconf_advanced_basic"),
            InlineKeyboardButton("None+Adv", callback_data="tavconf_none_advanced"),
            InlineKeyboardButton("Basic+Adv", callback_data="tavconf_basic_advanced"),
            InlineKeyboardButton("Adv+Adv", callback_data="tavconf_advanced_advanced")
        ]
        rows = [buttons[i:i + 3] for i in range(0, len(buttons), 3)]
        rows.append([InlineKeyboardButton(get_text(user_id, 'button_cancel'), callback_data="cancel")])
        await query.edit_message_text(get_text(user_id, 'tavily_depth_prompt'), reply_markup=InlineKeyboardMarkup(rows), parse_mode='HTML')
        return
    if data.startswith('tavconf_'):
        parts = data.split('_')
        config_str = f"{parts[1]}_{parts[2]}"
        set_user_data(user_id, 'web_search_mode', 'tavily')
        set_user_data(user_id, 'tavily_config', config_str)
        display = f"{parts[1].title()}+{parts[2].title()}"
        await notify_and_close('ws_tavily_updated', mode=display)
        return
    if data == 'ws_mode_compound':
        set_user_data(user_id, 'web_search_mode', 'compound')
        await notify_and_close('ws_compound_updated')
        return
    if data == 'ws_mode_off':
        set_user_data(user_id, 'web_search_mode', 'off')
        await notify_and_close('ws_off_updated')
        return

    # ... (Other settings callbacks) ...
    if data.startswith('set_model_'):
        set_user_data(user_id, 'selected_model_key', data.split('_')[2])
        m_name = MODELS[data.split('_')[2]]['name']
        await notify_and_close('model_updated', model_name=m_name)
        return
    if data.startswith('set_lang_'):
        set_user_data(user_id, 'language', data.split('_')[2])
        await notify_and_close('language_updated')
        return
    if data.startswith('tok_'):
        set_user_data(user_id, 'max_completion_tokens', int(data.split('_')[1]))
        await notify_and_close('max_tokens_updated', tokens=data.split('_')[1])
        return
    if data.startswith('tg_'):
        set_user_data(user_id, 'use_telegraph', data.split('_')[1])
        await notify_and_close('tg_updated', mode=data.split('_')[1])
        return
    if data == 'erase_confirm':
        delete_user_data(user_id)
        await notify_and_close('data_erased_confirm')
        return

    # ... (Prompt/API Input state setup) ...
    state_map = {
        'set_api_groq_main': 'awaiting_groq_api_key',
        'set_api_groq_fallback': 'awaiting_groq_fallback_api_key',
        'set_api_tavily_main': 'awaiting_tavily_api_key',
        'set_api_tavily_fallback': 'awaiting_tavily_fallback_api_key',
        'set_prompt_system': 'awaiting_system_prompt',
        'set_prompt_image': 'awaiting_image_prompt',
        'set_prompt_web': 'awaiting_websearch_prompt'
    }
    if data in state_map:
        target_state = state_map[data]
        set_user_data(user_id, 'state', target_state)
        db_key = target_state.replace('awaiting_', '')
        current_val = get_user_data(user_id, db_key, '')
        display_val = mask_api_key(current_val) if 'api_key' in db_key else escape_html(current_val)
        key_trans = target_state.replace('awaiting_', '') + "_prompt"
        msg_text = get_text(user_id, key_trans, prompt=display_val, masked_key=display_val)
        if "MISSING_KEY" in msg_text: msg_text = f"Current:\n<code>{display_val}</code>\nSend new value:"
        markup = InlineKeyboardMarkup([[InlineKeyboardButton(get_text(user_id, 'button_cancel'), callback_data="cancel")]])
        await query.edit_message_text(msg_text, reply_markup=markup, parse_mode='HTML')

# --- MAIN MESSAGE HANDLER ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    text = update.message.text or update.message.caption or ""
    state = get_user_data(user_id, 'state')
    
    if text.startswith('/'):
        await delete_trigger_message(update)
        return

    if state and state != 'chatting':
        # Input handling logic (API keys, Prompts, Temp)
        # ... (Same logic as before, handled inputs and clears state) ...
        await delete_trigger_message(update)
        cfg_msg_id = get_user_data(user_id, 'config_msg_id')
        if cfg_msg_id:
            try: await context.bot.delete_message(chat_id=user_id, message_id=cfg_msg_id)
            except: pass
        set_user_data(user_id, 'config_msg_id', None)

        if state == 'awaiting_temperature':
            try:
                val = float(text)
                if 0.0 <= val <= 2.0:
                    set_user_data(user_id, 'temperature', val)
                    set_user_data(user_id, 'state', 'chatting')
                    await send_ephemeral(context, user_id, f"✅ Temp: {val}")
                    await update_bot_menu(user_id, context)
                else:
                    await send_ephemeral(context, user_id, get_text(user_id, 'invalid_value'))
            except ValueError:
                await send_ephemeral(context, user_id, get_text(user_id, 'invalid_format'))
            return

        key_map = {
            'awaiting_groq_api_key': 'groq_api_key',
            'awaiting_groq_fallback_api_key': 'groq_fallback_api_key',
            'awaiting_tavily_api_key': 'tavily_api_key',
            'awaiting_tavily_fallback_api_key': 'tavily_fallback_api_key',
            'awaiting_system_prompt': 'system_prompt',
            'awaiting_image_prompt': 'image_prompt',
            'awaiting_websearch_prompt': 'websearch_prompt'
        }
        if state in key_map:
            set_user_data(user_id, key_map[state], text.strip())
            set_user_data(user_id, 'state', 'chatting')
            await send_ephemeral(context, user_id, "✅ Saved.")
            await update_bot_menu(user_id, context)
            return

    # Chat Logic
    content_payload = text
    log_activity(user_id, f"Text Input: {text[:30]}...")
    history = get_user_data(user_id, 'current_chat', [])

    # 1. Audio
    if update.message.voice or update.message.audio:
        await context.bot.send_chat_action(chat_id=user_id, action='typing')
        try:
            file_obj = await (update.message.voice or update.message.audio).get_file()
            audio_io = BytesIO()
            await file_obj.download_to_memory(out=audio_io)
            audio_io.seek(0)
            trans = await transcribe_audio(user_id, audio_io.read())
            content_payload = f"[Audio]: {trans.text}\n{text}"
        except Exception:
            await send_ephemeral(context, user_id, get_text(user_id, 'audio_error'))
            return

    # 2. Image
    if update.message.photo:
        await context.bot.send_chat_action(chat_id=user_id, action='typing')
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = bytes(await photo_file.download_as_bytearray())
        
        model_key = get_user_data(user_id, 'selected_model_key', DEFAULT_MODEL_KEY)
        model_type = MODELS[model_key]['type']
        
        if model_type not in ['multimodal', 'compound']:
            # ONE-OFF Description for text models
            image_prompt = get_user_data(user_id, 'image_prompt', get_text(user_id, 'default_image_prompt'))
            b64_img = base64.b64encode(photo_bytes).decode('utf-8')
            
            # Use Random API Key for the description
            desc_api_key = get_random_api_key(user_id, 'groq')
            if desc_api_key:
                client = Groq(api_key=desc_api_key)
                try:
                    resp = await asyncio.to_thread(client.chat.completions.create, 
                        model=MODELS['llama4']['id'],
                        messages=[{"role": "user", "content": [{"type": "text", "text": image_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}]}],
                        max_tokens=800
                    )
                    desc = resp.choices[0].message.content
                    content_payload = f"[Image Analysis]: {desc}\n{content_payload}"
                except Exception:
                    await send_ephemeral(context, user_id, get_text(user_id, 'photo_error'))
                    return
            else:
                await send_ephemeral(context, user_id, "No Groq API Key.")
                return
        else:
            content_payload = [{"type": "text", "text": content_payload or "Analyze this image."}, {"type": "image_bytes", "bytes": photo_bytes}]

    # 3. Web Search Logic
    ws_mode = get_user_data(user_id, 'web_search_mode', 'off')
    selected_model_key = get_user_data(user_id, 'selected_model_key', DEFAULT_MODEL_KEY)
    is_native_compound = (MODELS[selected_model_key]['type'] == 'compound')

    # Convert complex content to string for search query gen if needed
    search_input_text = content_payload
    if isinstance(content_payload, list):
        search_input_text = " ".join([x['text'] for x in content_payload if x['type'] == 'text'])

    ephemeral_search_context = ""

    if ws_mode == 'tavily' and not is_native_compound and isinstance(content_payload, str):
        await context.bot.send_chat_action(chat_id=user_id, action='typing')
        search_params = await generate_search_params(user_id, history, search_input_text)
        t_config = get_user_data(user_id, 'tavily_config', 'simple_basic')
        res = await perform_tavily_search(user_id, search_params, config=t_config)
        ephemeral_search_context = f"\n\n--- Web Search Results ---\n{res}"

    elif ws_mode == 'compound' and not is_native_compound and isinstance(content_payload, str):
        await context.bot.send_chat_action(chat_id=user_id, action='typing')
        params = await generate_search_params(user_id, history, search_input_text)
        query_str = params.get("query", search_input_text)
        res = await perform_compound_search(user_id, query_str)
        ephemeral_search_context = f"\n\n--- Search Results ---\n{res}"

    # 4. Save CLEAN User message to history (No search results yet)
    history.append({"role": "user", "content": content_payload, "message_id": update.message.message_id})
    set_user_data(user_id, 'current_chat', history)

    # 5. Prepare Payload & Inject Context
    # Note: If Native Compound is selected (is_native_compound), we just send the message.
    # The model handles search internally.
    
    if ws_mode == 'compound' and is_native_compound:
        # Native mode, do nothing special here
        pass

    # Build messages from CLEAN history
    messages = await prepare_messages(user_id, history, selected_model_key)
    
    # INJECT EPHEMERAL CONTEXT into the last message of the payload
    if ephemeral_search_context:
        last_msg = messages[-1]
        if isinstance(last_msg['content'], str):
            last_msg['content'] += ephemeral_search_context
        elif isinstance(last_msg['content'], list):
            for part in last_msg['content']:
                if part['type'] == 'text':
                    part['text'] += ephemeral_search_context
                    break

    user_tok = get_user_data(user_id, 'max_completion_tokens', DEFAULT_MAX_TOKENS)
    model_ctx = MODELS[selected_model_key]['context']
    actual_tokens = min(user_tok, model_ctx // 2)

    payload = {
        "model": MODELS[selected_model_key]['id'],
        "messages": messages,
        "temperature": float(get_user_data(user_id, 'temperature', 0.6)),
        "max_tokens": actual_tokens
    }

    if MODELS[selected_model_key]['type'] == 'reasoner': 
        payload['reasoning_effort'] = 'medium'
    
    if MODELS[selected_model_key]['type'] == 'compound': 
        payload['compound_custom'] = {"tools": {"enabled_tools": ["web_search", "code_interpreter"]}}

    await context.bot.send_chat_action(chat_id=user_id, action='typing')
    
    try:
        response = await execute_groq_request(user_id, payload)
        bot_text = strip_think_tags(response.choices[0].message.content) or "[Empty Response]"
        
        sent_msg = await send_response(update, context, bot_text)
        bot_id = sent_msg.message_id if sent_msg else None
        
        history.append({"role": "assistant", "content": bot_text, "message_id": bot_id})
        set_user_data(user_id, 'current_chat', history)

    except ValueError:
        await send_ephemeral(context, user_id, "❌ Invalid API Key.", 5)
    except Exception as e:
        log_activity(user_id, f"Inference Error: {e}")
        # Silent fail or generic message
        await send_ephemeral(context, user_id, get_text(user_id, 'internal_error'), 3)