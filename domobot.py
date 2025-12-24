import logging
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler
from config import TELEGRAM_TOKEN
from localization import load_languages
from handlers import (
    start_command, generic_menu_command_handler, new_chat_command, delete_last_command,
    handle_message, button_callback
)
from utils import setup_logger

logger = setup_logger("domobot")

def main():
    load_languages()
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("new_chat", new_chat_command))
    app.add_handler(CommandHandler("delete_last", delete_last_command))
    
    # Generic Menu Commands
    app.add_handler(CommandHandler("models", lambda u, c: generic_menu_command_handler(u, c, 'models')))
    app.add_handler(CommandHandler("web_search", lambda u, c: generic_menu_command_handler(u, c, 'web_search')))
    app.add_handler(CommandHandler("prompt", lambda u, c: generic_menu_command_handler(u, c, 'prompt')))
    app.add_handler(CommandHandler("api", lambda u, c: generic_menu_command_handler(u, c, 'api')))
    app.add_handler(CommandHandler("use_telegraph", lambda u, c: generic_menu_command_handler(u, c, 'telegraph')))
    app.add_handler(CommandHandler("temperature", lambda u, c: generic_menu_command_handler(u, c, 'temperature')))
    app.add_handler(CommandHandler("max_completion_tokens", lambda u, c: generic_menu_command_handler(u, c, 'tokens')))
    app.add_handler(CommandHandler("language", lambda u, c: generic_menu_command_handler(u, c, 'language')))
    app.add_handler(CommandHandler("erase_me", lambda u, c: generic_menu_command_handler(u, c, 'erase')))

    app.add_handler(CallbackQueryHandler(button_callback))
    app.add_handler(MessageHandler(filters.ALL, handle_message))

    logger.info("Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    main()