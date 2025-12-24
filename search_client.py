import httpx
import json
import logging
from datetime import datetime, timezone

from database import get_user_data
from utils import get_random_api_key, log_activity
from groq_client import execute_groq_request
from localization import get_text
from config import TAVILY_API_URL, MODELS

logger = logging.getLogger(__name__)

async def generate_search_params(user_id: int, history: list, user_input: str) -> dict:
    """
    Uses Structured Outputs to generate clean search parameters.
    """
    log_activity(user_id, "Generating Structured Search Params")
    
    web_prompt = get_user_data(user_id, 'websearch_prompt', get_text(user_id, 'default_websearch_prompt'))
    current_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    
    system_msg = (
        f"Current Date: {current_date}.\n"
        f"{web_prompt}\n"
        "Analyze the conversation and generate the optimal search parameters."
    )

    # Use short context for query generation to save tokens/time
    short_context = history[-3:]
    messages = [{"role": "system", "content": system_msg}]
    
    for msg in short_context:
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join([x.get("text", "") for x in content if x.get("type") == "text"])
        messages.append({"role": msg["role"], "content": content})
    
    messages.append({"role": "user", "content": user_input})

    # Strict JSON Schema
    schema = {
        "name": "tavily_search_params",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The optimized search query string"},
                "topic": {"type": "string", "enum": ["general", "news"]},
                "days": {"type": "integer", "default": 3},
                "country": {"type": ["string", "null"]}
            },
            "required": ["query", "topic", "days", "country"],
            "additionalProperties": False
        }
    }

    # Use GPT-OSS for reliable structured output
    payload = {
        "model": "openai/gpt-oss-120b", 
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 512,
        "response_format": {"type": "json_schema", "json_schema": schema}
    }
    
    try:
        response = await execute_groq_request(user_id, payload)
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logger.error(f"Structured Gen Failed: {e}")
        # Fallback parameters
        return {"query": user_input, "topic": "general", "days": 3, "country": None}

async def perform_compound_search(user_id: int, query: str) -> str:
    """
    Uses 'groq/compound' native tool to perform a search.
    This function acts as a Tool that returns text content.
    """
    log_activity(user_id, f"Compound Search Tool Query: {query}")
    
    payload = {
        "model": "groq/compound",
        "messages": [{"role": "user", "content": query}],
        "compound_custom": {
            "tools": {
                "enabled_tools": ["web_search"]
            }
        }
    }
    
    try:
        # We execute the request. Compound will do the search internally and return a synthesis.
        # We treat this synthesis as the "Search Result" context for the main model.
        response = await execute_groq_request(user_id, payload)
        return response.choices[0].message.content
    except Exception as e:
        log_activity(user_id, f"Compound Search Error: {e}")
        return "Error performing Compound Search."

async def perform_tavily_search(user_id: int, params: dict, config: str = "simple_basic") -> str:
    api_key = get_random_api_key(user_id, 'tavily')
    if not api_key: return "Error: No Tavily Key"

    try:
        include_answer, search_depth = config.split('_')
    except:
        include_answer, search_depth = "none", "basic"

    query_str = params.get("query")
    log_activity(user_id, f"Tavily Search: {query_str} | Mode: {include_answer}+{search_depth}")

    payload = {
        "api_key": api_key,
        "query": query_str,
        "topic": params.get("topic", "general"),
        "search_depth": search_depth,
        "include_answer": include_answer if include_answer != "none" else None,
        "max_results": 5,
        "days": params.get("days", 3)
    }
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(TAVILY_API_URL, json=payload, timeout=20.0)
            if resp.status_code == 200:
                data = resp.json()
                res_txt = ""
                
                if include_answer != "none" and data.get("answer"):
                    res_txt += f"{data['answer']}\n\n"
                
                res_txt += "Sources:\n"
                for res in data.get("results", []):
                    title = res.get('title', 'No Title')
                    url = res.get('url', '#')
                    content = res.get('content', '')[:300].replace("\n", " ")
                    res_txt += f"• <a href='{url}'>{title}</a>: {content}\n"
                
                return res_txt.strip()
            
            return f"Search Error: {resp.status_code}"
    except Exception as e:
        logger.error(f"Tavily Req Failed: {e}")
        return "Search Error: Connection Failed"