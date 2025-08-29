import asyncio
import logging
import base64
from datetime import datetime, timezone
from groq import Groq, RateLimitError, APIStatusError
from telegram import Update
from telegram.ext import ContextTypes

from bot.database import get_user_data, set_user_data
from bot.localization import get_text
from bot.utils import send_temp_message
from config import MODELS, DEFAULT_MODEL, VISION_MODELS

logger = logging.getLogger(__name__)

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

async def transcribe_audio_with_retry(update: Update, context: ContextTypes.DEFAULT_TYPE, audio_bytes: bytes):
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
            return await asyncio.to_thread(
                groq_client.audio.transcriptions.create,
                file=("audio.mp3", audio_bytes),
                model="whisper-large-v3"
            )
        except (RateLimitError, APIStatusError) as e:
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

    # CONTEXT WINDOW FIX: Limit the history to the last 20 messages
    short_history = history[-20:]

    groq_messages = [{"role": "system", "content": f"Today is {datetime.now(timezone.utc).strftime('%Y-%m-%d')}. {system_prompt}"}]
    is_multimodal = model_name in VISION_MODELS
    for message in short_history:
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