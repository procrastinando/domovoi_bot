import asyncio
import re
import logging
from datetime import datetime, timezone
from groq import Groq, RateLimitError, AuthenticationError, APIStatusError

from database import get_user_data, set_user_data
from utils import log_activity
from localization import get_text
from config import MODELS

# Helper to get all available keys for rotation
def get_available_keys(user_id):
    k1 = get_user_data(user_id, 'groq_api_key')
    k2 = get_user_data(user_id, 'groq_fallback_api_key')
    return [k for k in [k1, k2] if k and k.strip()]

async def execute_groq_request(user_id: int, payload: dict):
    """
    Orchestrates API calls with:
    1. Key Rotation (Main -> Fallback)
    2. Smart Waiting (Respects retry-after)
    3. Impossible Request Handling (Prunes if Request > Limit)
    """
    keys = get_available_keys(user_id)
    if not keys:
        raise ValueError("No Groq API Key")

    # We try indefinitely until MAX_TOTAL_RETRIES or success
    # But to prevent infinite loops, we cap total attempts
    MAX_TOTAL_ATTEMPTS = 10 
    attempt = 0
    
    # Track history locally to prune without mutating DB until necessary
    # Note: prepare_messages already built the payload. 
    # If we need to prune, we must manipulate payload['messages'].
    
    while attempt < MAX_TOTAL_ATTEMPTS:
        # Rotate keys based on attempt number
        current_key = keys[attempt % len(keys)]
        client = Groq(api_key=current_key, max_retries=0)
        
        try:
            return await asyncio.to_thread(client.chat.completions.create, **payload)

        except AuthenticationError:
            log_activity(user_id, "Groq Auth Error: Invalid Key. Trying next key/retrying.")
            # If all keys are invalid, this will loop until max attempts. 
            # In production, maybe mark key as bad.
            attempt += 1
            continue

        except (RateLimitError, APIStatusError) as e:
            # Check for Rate Limit (429) or Context/TPM Limit (413)
            is_limit = isinstance(e, RateLimitError) or (isinstance(e, APIStatusError) and e.status_code == 413)
            
            if not is_limit:
                raise e # 500 errors, etc.

            # Analyze Error Message for TPM vs Request Size
            # Error msg format: "Limit 10000, Requested 10702"
            error_msg = str(e)
            match = re.search(r"Limit (\d+), Requested (\d+)", error_msg)
            
            must_prune = False
            
            if match:
                limit = int(match.group(1))
                requested = int(match.group(2))
                if requested > limit:
                    # Waiting won't help. The request is physically too big for the tier.
                    log_activity(user_id, f"Request ({requested}) > Limit ({limit}). Pruning immediately.")
                    must_prune = True
            
            # Get Retry Time
            retry_after = 2.0 # Default
            try:
                if hasattr(e, 'response') and e.response.headers.get('retry-after'):
                    retry_after = float(e.response.headers.get('retry-after'))
                # Sometimes it's in the text: "Please try again in 4s"
                elif "try again in" in error_msg:
                    sec_match = re.search(r"in (\d+(\.\d+)?)s", error_msg)
                    if sec_match:
                        retry_after = float(sec_match.group(1))
            except:
                pass

            # STRATEGY DECISION
            
            # 1. If we have another key and haven't just switched, try switching first
            #    (Unless we MUST prune because request > limit)
            if not must_prune and len(keys) > 1 and (attempt % len(keys)) != (len(keys) - 1):
                log_activity(user_id, "Rate limit hit. Swapping API Key immediately.")
                attempt += 1
                continue

            # 2. If we MUST prune (Impossible Request), do it now
            if must_prune:
                # Pruning logic
                msgs = payload.get('messages', [])
                # [0] System, [1] User, [2] Bot...
                # We need to remove 2 messages from history (indices 1 and 2)
                if len(msgs) > 3:
                    msgs.pop(1)
                    msgs.pop(1)
                    payload['messages'] = msgs
                    
                    # Update DB to reflect pruning
                    history = get_user_data(user_id, 'current_chat', [])
                    if len(history) >= 2:
                        history.pop(0)
                        if len(history) > 0: history.pop(0)
                        set_user_data(user_id, 'current_chat', history)
                    
                    log_activity(user_id, "Pruned 2 messages. Retrying immediately.")
                    attempt += 1
                    continue
                else:
                    # Can't prune anymore, giving up
                    log_activity(user_id, "Cannot prune further. Request too large.")
                    raise e

            # 3. If we are here, we need to WAIT.
            log_activity(user_id, f"Rate limit hit on all keys. Sleeping {retry_after}s...")
            await asyncio.sleep(retry_after)
            
            # After waiting, we increment attempt and try again (same key or next key)
            attempt += 1
            continue

    raise Exception("Max retries exceeded for API request.")

async def transcribe_audio(user_id: int, audio_bytes: bytes):
    # Simple rotation for audio too
    keys = get_available_keys(user_id)
    if not keys: raise ValueError("No API Key")
    
    # Try first key, if fail try second
    for key in keys:
        try:
            client = Groq(api_key=key)
            return await asyncio.to_thread(
                client.audio.transcriptions.create,
                file=("audio.mp3", audio_bytes),
                model="whisper-large-v3"
            )
        except Exception as e:
            log_activity(user_id, f"Audio Transcribe Error on key ...{key[-4:]}: {e}")
            continue
    raise Exception("Audio transcription failed on all keys")

async def prepare_messages(user_id: int, history: list, model_key: str):
    sys_prompt = get_user_data(user_id, 'system_prompt', get_text(user_id, 'default_system_prompt'))
    messages = [{"role": "system", "content": f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}. {sys_prompt}"}]
    
    model_info = MODELS.get(model_key, MODELS['llama4'])
    is_multimodal = model_info['type'] in ['multimodal', 'compound']

    # Keep web search results ONLY for the last 2 user messages (current + previous)
    user_indices = [i for i, x in enumerate(history) if x['role'] == 'user']
    if len(user_indices) > 2:
        cutoff_index = user_indices[-2]
    else:
        cutoff_index = 0

    # Strict Regex for cleaning old search results
    # Matches: newline + newline + --- Search Results (Any) --- + everything after
    # OR the specific format used in handlers.py
    
    for i, msg in enumerate(history):
        msg_copy = msg.copy()
        content = msg_copy.get('content', '')
        role = msg_copy['role']
        
        # 1. Clean Old Web Search Results
        # If this is a user message AND it is older than our cutoff
        if role == 'user' and i < cutoff_index and isinstance(content, str):
            # Look for the separator used in handlers.py
            # "--- Web Search Results ---" or "--- Search Results (Compound) ---" etc.
            if "\n\n---" in content:
                # Split and take the first part (original user query)
                clean_content = content.split("\n\n---")[0]
                msg_copy['content'] = clean_content.strip()

        # 2. Handle Multimodal Images
        if isinstance(content, list):
            if is_multimodal:
                new_content = []
                for item in content:
                    if item.get("type") == "image_bytes":
                        b64 = base64.b64encode(item["bytes"]).decode('utf-8')
                        new_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                    else:
                        new_content.append(item)
                msg_copy["content"] = new_content
            else:
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                msg_copy["content"] = "\n".join(text_parts)
        
        for k in ['timestamp', 'message_id']:
            msg_copy.pop(k, None)
        
        messages.append(msg_copy)
        
    return messages