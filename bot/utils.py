import re
import asyncio
import logging
import httpx
from datetime import datetime, timezone
from telegram import Update
from telegram.ext import ContextTypes
from groq import Groq
from telegraph.aio import Telegraph
from markdown_it import MarkdownIt

from bot.database import get_user_data, set_user_data
from config import TELEGRAM_MSG_LIMIT, TAVILY_API_URL, TAVILY_USAGE_URL

logger = logging.getLogger(__name__)
telegraph = Telegraph()

def escape_markdown_v2(text: str) -> str:
    """
    Escapes characters for MarkdownV2. 
    Used primarily for the final response generation, not for command menus anymore.
    """
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

def escape_html(text: str) -> str:
    """
    Escapes characters for HTML parsing.
    Used for command menus to ensure clean text and copyable code blocks.
    """
    if not text: return ""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

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
        logger.warning(f"Groq API key validation failed: {e}")
        return False

async def is_tavily_key_valid_sync(api_key: str) -> bool:
    if not api_key or not api_key.startswith("tvly-"):
        return False
    # Validate by making a small request
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            payload = {"query": "test", "max_results": 1}
            response = await client.post(TAVILY_API_URL, headers=headers, json=payload, timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False

async def get_tavily_usage(api_key: str) -> str:
    """
    Fetches usage stats from Tavily API. Returns a string 'used/limit' or 'N/A'.
    """
    if not api_key:
        return "N/A"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                TAVILY_USAGE_URL, 
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=5.0
            )
            if response.status_code == 200:
                data = response.json()
                account = data.get('account', {})
                usage = account.get('plan_usage', 0)
                limit = account.get('plan_limit', '∞')
                return f"{usage}/{limit}"
    except Exception as e:
        logger.error(f"Error fetching Tavily usage: {e}")
    
    return "Error"

async def perform_web_search(user_id: int, query: str, max_results: int = 5):
    """
    Executes a Tavily search with failover/swapping logic.
    Requires user_id to access and swap keys in the database.
    """
    logger.info(f"Performing Tavily web search for user {user_id}: {query}")
    attempt_count = 0
    max_swaps = 1

    while True:
        # Always fetch the current main key
        api_key = get_user_data(user_id, 'tavily_api_key')
        if not api_key:
            return "Error: Main Tavily API Key not set."

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "query": query,
            "search_depth": "basic",
            "topic": "general",
            "max_results": max_results,
            "include_answer": "advanced",
            "include_raw_content": False,
            "include_images": False
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(TAVILY_API_URL, headers=headers, json=payload, timeout=15.0)
                
                # Check for success
                if response.status_code == 200:
                    data = response.json()
                    parts = []
                    if data.get("answer"):
                        parts.append(f"--- Search Summary (Tavily AI) ---\n{data['answer']}")
                    
                    parts.append(f"\n--- Search Results for '{data.get('query', query)}' ---")
                    for i, res in enumerate(data.get("results", []), 1):
                        parts.append(f"[{i}] {res.get('title', 'N/A')}\nLink: {res.get('url', 'N/A')}\nSnippet: {res.get('content', '')}")
                    
                    return "\n\n".join(parts) if parts else "No web search results found."

                # Handle Failures that trigger a swap (Auth errors, Rate limits)
                if response.status_code in [401, 403, 429]:
                    fallback_key = get_user_data(user_id, 'tavily_fallback_api_key')
                    if fallback_key and attempt_count < max_swaps:
                        attempt_count += 1
                        logger.warning(f"Tavily Main key failed ({response.status_code}). Swapping to fallback.")
                        
                        # --- PERMANENT SWAP LOGIC ---
                        set_user_data(user_id, 'tavily_api_key', fallback_key)
                        set_user_data(user_id, 'tavily_fallback_api_key', api_key)
                        continue # Retry immediately with new main key

                    logger.error(f"Tavily API failed {response.status_code} and no fallback available/working.")
                    return f"Search Error: API returned {response.status_code}. Please check keys or limits."
                
                # Other errors
                return f"Search Error: {response.status_code}"

        except Exception as e:
            logger.error(f"Error during Tavily web search: {e}")
            return "Search Error: Connection failed."
        
        # Break loop if we get here naturally
        break
    return None