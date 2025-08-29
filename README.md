# Groq Multimodal Telegram Bot

A powerful, feature-rich, and multi-language Telegram bot that leverages the blazing-fast Groq API for conversational AI. This bot is designed to be a personal AI assistant, complete with persistent memory, multimodal capabilities (text, images, and audio), web search integration, and deep user customization for performance and output.

[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-green?style=flat-square)](https://groq.com/)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://www.python.org/)
[![Multi-language](https://img.shields.io/badge/Language-Multi-orange?style=flat-square)](#-features)

## ✨ Features

*   🌍 **Multi-language Support:**
    *   Automatically detects the user's language on the first start.
    *   Fully internationalized interface with support for 10+ languages via a simple `language.yaml` file.
    *   Users can switch languages at any time with the `/language` command.

*   🧠 **High-Performance AI:** Engage in text-based conversations using powerful models like Llama 4 Maverick and GPT-Oss 120B on the Groq LPU™ Inference Engine.

*   🖼️ **Intelligent Image Understanding:** Send a photo, and the bot will analyze it. If your current model doesn't support images, it intelligently uses a vision model for analysis and injects the description into your ongoing chat.

*   🗣️ **Voice Transcription:** Transcribes voice notes using Whisper-Large-V3, then responds to the transcribed text.

*   🌐 **Web Search Integration:** A toggleable web search mode provides the bot with real-time information from DuckDuckGo to answer questions about current events.

*   ⚙️ **Fine-Grained Performance Control:**
    *   `/models`: Switch between AI models and set special parameters like **reasoning effort** for GPT-Oss.
    *   `/temperature`: Adjust the model's creativity and randomness (0.0 to 1.0).
    *   `/max_completion_tokens`: Set a preferred maximum response length, which automatically adapts to the hard limits of the currently selected model.

*   📄 **Advanced Output Handling with Telegra.ph:**
    *   `/use_telegraph`: Choose how the bot handles messages that exceed Telegram's character limit: split them, or automatically post them as clean, readable [Telegra.ph](https://telegra.ph/) articles.

*   🛠️ **Dynamic & Clean User Interface:**
    *   The bot's command menu (`/`) updates instantly to show your current settings for the model, temperature, token limits, and more, **all in your selected language**.
    *   Commands are deleted after use to keep the chat history uncluttered.

*   💾 **Persistent & Private User Settings:**
    *   Remembers each user's API keys, model choice, language, custom prompts, and all preferences in a local `database.yaml` file.
    *   `/erase_me`: Users can permanently delete all their data from the bot's database at any time.

*   🔔 **Robust Error Handling & API Management:**
    *   `/api` & `/fallback_api`: Set both a primary and a backup Groq API key. The bot will automatically swap to the fallback key if the primary one fails (e.g., due to rate limits).
    *   Notifies you in-chat when an API rate limit is hit and automatically retries.
    *   Warns you if the conversation history becomes too long and suggests a solution.

## How to Run

### 1. Using Docker Compose (Recommended)

The easiest way to get started is with Docker.

**Setup:**

1.  Create a `.env` file in the same directory and add your Telegram token:
    ```
    TELEGRAM_TOKEN="YOUR_TELEGRAM_BOT_TOKEN_HERE"
    ```
2.  Download the `language.yaml` file from the repository and place it in the same directory.
3.  Create a `docker-compose.yml` file with the following content:

```yaml
services:
  domovoi_bot:
    # build from source if you have custom changes:
    build:
      context: https://github.com/procrastinando/domovoi_bot.git#main
    container_name: domovoi_bot
    env_file:
      - .env
    volumes:
      # Mounts the local database and language file into the container
      - domovoi_bot_data:/app
      - ./language.yaml:/app/language.yaml
    restart: always

volumes:
  domovoi_bot_data:
```

**Run:**

```bash
docker-compose up -d
```

### 2. Manual Installation

#### Prerequisites

*   Python 3.10+
*   [FFmpeg](https://ffmpeg.org/download.html) installed and available in your system's PATH (for audio conversion).
*   A **Groq API Key** (from [console.groq.com](https://console.groq.com/keys)).
*   A **Telegram Bot Token** (from [@BotFather](https://t.me/BotFather)).

#### Installation and Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/procrastinando/domovoi_bot.git
    cd domovoi_bot
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Required Libraries**
    The required Python packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    The bot loads your Telegram token from an environment variable for security.
    ```bash
    export TELEGRAM_TOKEN="YOUR_TELEGRAM_BOT_TOKEN_HERE"
    ```
    (On Windows, use `set TELEGRAM_TOKEN="YOUR_TOKEN"`)

5.  **Run the Bot**
    Make sure the `language.yaml` file is in the same directory as your script.
    ```bash
    python domobot.py # Or your script's filename
    ```

## Usage

1.  **Start a Chat:** Open Telegram and find your bot. Send the `/start` command.
2.  **Provide API Key:** The bot will prompt you for your Groq API key. Paste it into the chat. This is a one-time setup.
3.  **Interact:** You can now chat with the bot!
    *   Send a text message.
    *   Send a photo (with an optional caption).
    *   Send a voice message or an audio file.
4.  **Use Commands:** Use the menu button next to the text input field or type the commands directly to configure the bot.

### Available Commands

The command descriptions in your Telegram menu update dynamically to reflect your current settings and language!

| Command                   | Description                                                      |
| :------------------------ | :--------------------------------------------------------------- |
| `/start`                  | Initializes the bot, detects your language, and sets up your profile. |
| `/new_chat`               | Clears the current conversation history and starts a fresh chat. |
| `/language`               | Choose a new interface language for the bot.                     |
| `/models`                 | Choose from available AI models (with special options).          |
| `/temperature`            | Set the model's creativity/randomness (0.0 to 2.0).              |
| `/max_completion_tokens`  | Set the max response length. Adapts to model limits.             |
| `/web_search`             | Toggle real-time web search ON or OFF.                           |
| `/use_telegraph`          | Control how long messages are handled (Split or Telegra.ph).     |
| `/system_prompt`          | Set a custom personality or instruction set for the bot.         |
| `/image_prompt`           | Set the prompt used for image analysis by the vision model.      |
| `/api`                    | View your masked primary Groq API key or update it.              |
| `/fallback_api`           | Set a backup Groq API key for automatic failover.                |
| `/erase_me`               | Permanently delete all your data (keys, settings, current chat). |