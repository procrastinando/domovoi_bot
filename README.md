# Groq Multimodal Telegram Bot

A powerful, feature-rich Telegram bot that leverages the blazing-fast Groq API for conversational AI. This bot is designed to be a personal AI assistant, complete with persistent memory, multimodal capabilities (text, images, and audio), web search integration, and deep user customization for performance and output.

[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-green?style=flat-square)](https://groq.com/)

## ✨ Features

*   🧠 **High-Performance AI:** Engage in text-based conversations using powerful models like Llama 4 Maverick and GPT-Oss 120B on the Groq LPU™ Inference Engine.
*   🖼️ **Intelligent Image Understanding:** Send a photo, and the bot will analyze it. If your current model doesn't support images, it will intelligently use a vision model for analysis and inject the description into your ongoing chat.
*   🗣️ **Audio & Voice Transcription:** Transcribes audio files and voice notes using Whisper-Large-V3, then responds to the transcribed text.
*   🌐 **Web Search Integration:** A toggleable web search mode provides the bot with real-time information from DuckDuckGo to answer questions about current events.
*   ⚙️ **Fine-Grained Performance Control:**
    *   `/models`: Switch between AI models and set special parameters like **reasoning effort** for GPT-Oss.
    *   `/temperature`: Adjust the model's creativity and randomness (0.0 to 1.0).
    *   `/max_completion_tokens`: Set a preferred maximum response length, which automatically adapts to the hard limits of the currently selected model.
*   📄 **Advanced Output Handling with Telegra.ph:**
    *   `/use_telegraph`: Choose how the bot handles messages that exceed Telegram's character limit: split them, or automatically post them as clean, readable [Telegra.ph](https://telegra.ph/) articles.
    *   **Table Support:** The bot instructs the AI to format tables as ASCII art within code blocks, ensuring they render perfectly in Telegra.ph.
*   🛠️ **Dynamic & Clean User Interface:**
    *   The bot's command menu (`/`) updates instantly to show your current settings for the model, temperature, token limits, and more.
    *   Commands are deleted after use to keep the chat history uncluttered.
*   💾 **Persistent User Settings:** Remembers each user's API key, model choice, system prompt, and all preferences in a local `database.yaml` file.
*   🔔 **Robust Error Handling & User Notifications:**
    *   Notifies you in-chat when an API rate limit is hit and automatically retries.
    *   Warns you if the conversation history becomes too long for the model and suggests a solution.
    *   Provides ephemeral feedback if a web search fails.

## How to Run

### 1. As Docker Compose:

The easiest way to get started is with Docker.

```yaml
services:
  groq_bot:
    # Use a pre-built image or build from source
    image: procrastinando/domovoi_bot:latest
    # build:
    #   context: https://github.com/procrastinando/domovoi_bot.git#main
    container_name: groq_bot
    environment:
      # Define your secrets in an .env file or directly here
      - TELEGRAM_TOKEN=${TELEGRAM_TOKEN}
    volumes:
      - groq_bot_data:/app
    restart: always

volumes:
  groq_bot_data:
```

### 2. Manual Installation

#### Prerequisites

*   Python 3.10+
*   [FFmpeg](https://ffmpeg.org/download.html) installed and available in your system's PATH (required for audio conversion).
*   A **Groq API Key** (from [console.groq.com](https://console.groq.com/keys)).
*   A **Telegram Bot Token** (from [@BotFather](https://t.me/BotFather) on Telegram).

#### Installation and Setup

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Required Libraries**
    Create a `requirements.txt` file with the following content and run `pip install -r requirements.txt`.
    ```txt
    python-telegram-bot
    groq
    pyyaml
    ddgs
    telegraph[aio]
    markdown2
    ```

4.  **Configure Environment Variables**
    The bot loads your Telegram token from an environment variable for security. **Do not hardcode it in the script.**
    ```bash
    export TELEGRAM_TOKEN="YOUR_TELEGRAM_BOT_TOKEN_HERE"
    ```
    Alternatively, you can create a `.env` file in the project directory and use a library like `python-dotenv` to load it.

5.  **Run the Bot**
    Launch the bot from your terminal.
    ```bash
    python bot.py # Or your script's filename
    ```

## Usage

1.  **Start a Chat:** Open Telegram and find your bot. Send the `/start` command.
2.  **Provide API Key:** The bot will prompt you for your Groq API key. Paste it into the chat. This is a one-time setup per user.
3.  **Interact:** You can now chat with the bot!
    *   Send a text message.
    *   Send a photo (with an optional caption).
    *   Send a voice message or an audio file.
4.  **Use Commands:** Use the menu button next to the text input field or type the commands directly to configure the bot.

### Available Commands

The command descriptions in your Telegram menu update dynamically to reflect your current settings!

| Command | Description |
| :--- | :--- |
| `/start` | Initializes the bot and sets up your profile. |
| `/new_chat` | Archives the current conversation and starts a new one. |
| `/models` | Choose from a list of available AI models (with special options). |
| `/temperature` | Set the model's creativity/randomness (0.0 to 1.0). |
| `/max_completion_tokens` | Set the max response length. Adapts to model limits. |
| `/web_search` | Toggle real-time web search ON or OFF. |
| `/use_telegraph` | Control how long messages are handled (Split or Telegra.ph). |
| `/system_prompt` | Set a custom personality/instruction for the bot. |
| `/image_prompt` | Set the prompt used for image analysis by the vision model. |
| `/api` | View your masked Groq API key or update it. |
