import re
import asyncio
import logging
import base64
from datetime import datetime, timezone
from io import BytesIO
from telegram import Update
from telegram.ext import ContextTypes
from groq import Groq
from duckduckgo_search import DDGS
from telegraph.aio import Telegraph
from markdown_it import MarkdownIt

from bot.database import get_user_data, set_user_data
from config import TELEGRAM_MSG_LIMIT

logger = logging.getLogger(__name__)
telegraph = Telegraph()

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

async def perform_web_search(query: str):
    logger.info(f"Performing web search for: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        return "\n\n--- Web Search Results ---\n" + "\n\n".join([f"Title: {r.get('title', 'N/A')}\nSnippet: {r.get('body', 'N/A')}" for r in results]) if results else "No web search results found."
    except Exception as e:
        logger.error(f"Error during web search: {e}")
        return None