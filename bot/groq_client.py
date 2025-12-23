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
    # Allow 1 swap. If the new key also fails, we stop to avoid infinite loops.
    max_swaps = 1 

    while True:
        # Always load the current 'Main' key from the DB. 
        # If a swap happened in the previous iteration, this picks up the NEW main key.
        api_key = get_user_data(user_id, 'groq_api_key')
        
        if not api_key:
            await update.message.reply_text(get_text(user_id, 'api_key_not_set'))
            raise ValueError("Main API key not found.")
            
        groq_client = Groq(api_key=api_key, max_retries=0)
        
        try:
            params = {
                "model": model_config['name'], "temperature": float(model_config.get('temperature', 1.0)),
                "max_tokens": effective_tokens, "messages": messages
            }
            if 'reasoning_effort' in model_config:
                params['reasoning_effort'] = model_config['reasoning_effort']
                
            return await asyncio.to_thread(groq_client.chat.completions.create, **params)
            
        except (RateLimitError, APIStatusError) as e:
            # 1. Handle Context Window Errors (Swapping won't help here)
            if isinstance(e, APIStatusError) and e.status_code == 413:
                await update.message.reply_text(get_text(user_id, 'context_too_long'))
                raise

            # 2. Handle Rate Limits or other API Errors -> Attempt Swap
            fallback_key = get_user_data(user_id, 'fallback_api_key')
            
            # If we have a fallback and haven't swapped yet in this specific request cycle
            if fallback_key and attempt_count < max_swaps:
                attempt_count += 1
                logger.warning(f"Main API key failed for user {user_id}. PERFORMING PERMANENT SWAP.")
                
                # Notify user briefly
                await send_temp_message(context, user_id, get_text(user_id, 'api_key_fallback_attempt'), 2)
                
                # --- PERMANENT SWAP LOGIC ---
                # The working fallback becomes the new Main. The broken Main becomes the Fallback.
                set_user_data(user_id, 'groq_api_key', fallback_key)
                set_user_data(user_id, 'fallback_api_key', api_key)
                
                # Continue the 'while' loop. 
                # The next iteration calls get_user_data, which loads the NEW Main key.
                continue

            # 3. If we are here, either:
            #    a) No fallback exists
            #    b) We already swapped, and the fallback failed too (Both exhausted)
            
            if isinstance(e, RateLimitError):
                retry_after = int(e.response.headers.get("retry-after", 60))
                logger.warning(f"All available API keys failed rate limit for user {user_id}. Retrying after {retry_after}s.")
                await update.effective_chat.send_message(get_text(user_id, 'rate_limit_message', seconds=retry_after))
                await asyncio.sleep(retry_after)
                # After sleeping, we loop again to retry with the CURRENT main key
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
    max_swaps = 1

    while True:
        api_key = get_user_data(user_id, 'groq_api_key')
        if not api_key:
            await update.message.reply_text(get_text(user_id, 'api_key_not_set'))
            raise ValueError("Main API key not found.")
            
        groq_client = Groq(api_key=api_key, max_retries=0)
        
        try:
            return await asyncio.to_thread(
                groq_client.audio.transcriptions.create,
                file=("audio.mp3", audio_bytes),
                model="whisper-large-v3"
            )
        except (RateLimitError, APIStatusError) as e:
            fallback_key = get_user_data(user_id, 'fallback_api_key')
            
            if fallback_key and attempt_count < max_swaps:
                attempt_count += 1
                logger.warning(f"Main API key failed for user {user_id}. PERFORMING PERMANENT SWAP.")
                await send_temp_message(context, user_id, get_text(user_id, 'api_key_fallback_attempt'), 2)
                
                # --- PERMANENT SWAP LOGIC ---
                set_user_data(user_id, 'groq_api_key', fallback_key)
                set_user_data(user_id, 'fallback_api_key', api_key)
                continue

            if isinstance(e, RateLimitError):
                retry_after = int(e.response.headers.get("retry-after", 60))
                logger.warning(f"All available API keys failed rate limit for user {user_id}. Retrying after {retry_after}s.")
                await update.effective_chat.send_message(get_text(user_id, 'rate_limit_message', seconds=retry_after))
                await asyncio.sleep(retry_after)
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

async def generate_search_query(update: Update, context: ContextTypes.DEFAULT_TYPE, history: list, user_input: str):
    """
    Uses the LLM to generate a specific search query based on chat history.
    """
    user_id = update.effective_user.id
    
    system_msg = {
        "role": "system", 
        "content": (
            "You are a search query generator. Your task is to look at the user's last message "
            "and the conversation history, then output a SINGLE, precise web search query "
            "that will fetch the information the user needs. "
            "Do NOT answer the question. Do NOT explain. Output ONLY the query text."
            f"Current Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}."
            "Example:\n"
            "History: [User: most holidays in world? Bot: list...]\n"
            "User: how about portugal?\n"
            "Output: public holidays in Portugal 2025 statistics"
        )
    }

    short_history = history[-6:] 
    messages = [system_msg]
    
    for msg in short_history:
        content = msg.get("content")
        if isinstance(content, list):
             content = " ".join([item.get("text", "") for item in content if item.get("type") == "text"])
        messages.append({"role": msg["role"], "content": content})
    
    messages.append({"role": "user", "content": user_input})

    model_config = get_user_data(user_id, 'selected_model', DEFAULT_MODEL)
    
    try:
        completion = await get_groq_completion_with_retry(update, context, messages, model_config, effective_tokens=100)
        query = completion.choices[0].message.content.strip()
        query = query.replace('"', '').replace("Search query:", "").strip()
        logger.info(f"Generated search query for user {user_id}: {query}")
        return query
    except Exception as e:
        logger.error(f"Failed to generate search query: {e}")
        return user_input