import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler

from config import TELEGRAM_TOKEN
from bot.localization import load_languages
from bot.handlers import (
    start, new_chat, groq_api_command, groq_fallback_api_command,
    tavily_api_command, tavily_fallback_api_command,
    models_command, system_prompt_command, image_prompt_command, temperature_command,
    web_search_command, erase_me_command, max_completion_tokens_command,
    use_telegraph_command, language_command, button_callback, handle_user_input,
    handle_unsupported_message, handle_unknown_command
)

# --- Logging Setup ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def main() -> None:
    if not TELEGRAM_TOKEN:
        logger.error("Configuration error: TELEGRAM_TOKEN not set.")
        return
    load_languages()
    logger.info("Starting bot...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    cmd_handlers = {
        "start": start,
        "new_chat": new_chat,
        "groq_api": groq_api_command,             # Renamed
        "groq_fallback_api": groq_fallback_api_command, # Renamed
        "tavily_api": tavily_api_command,
        "tavily_fallback_api": tavily_fallback_api_command,
        "models": models_command,
        "system_prompt": system_prompt_command,
        "image_prompt": image_prompt_command,
        "temperature": temperature_command,
        "web_search": web_search_command,
        "erase_me": erase_me_command,
        "max_completion_tokens": max_completion_tokens_command,
        "use_telegraph": use_telegraph_command,
        "language": language_command
    }
    for cmd, hnd in cmd_handlers.items():
        app.add_handler(CommandHandler(cmd, hnd))

    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_input))
    app.add_handler(MessageHandler(filters.PHOTO, handle_user_input))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_user_input))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND & ~filters.TEXT & ~filters.PHOTO & ~filters.VOICE & ~filters.AUDIO, handle_unsupported_message))
    app.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))

    try:
        logger.info("Bot is polling...")
        app.run_polling()
    except Exception as e:
        logger.critical(f"Bot crashed: {e}", exc_info=True)

if __name__ == '__main__':
    main()