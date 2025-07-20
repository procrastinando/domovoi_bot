# Groq-Powered Multimodal Telegram Bot

A versatile, multimodal Telegram bot that uses the ultra-fast Groq API for text chat, image analysis, and audio transcription.

[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-green?style=flat-square)](https://groq.com/)

## Features

*   **Conversational AI:** Engage in text-based conversations using powerful models like Llama 4 Maverick.
*   **Image Understanding:** Send a photo, and the bot will analyze and describe its contents.
*   **Audio & Voice Transcription:** Transcribes audio files and voice notes using Whisper-Large-V3-Turbo, then responds to the transcribed text.
*   **Web Search Integration:** A toggleable web search mode provides the bot with real-time information to answer questions about current events.
*   **Dynamic Command Menu:** The bot's command menu updates automatically to reflect the current status of features like web search.
*   **On-the-Fly Configuration:**
    *   `/models`: Switch between different AI models.
    *   `/system_prompt`: Define the bot's personality and behavior.
    *   `/api`: Securely update your Groq API key.
*   **Persistent User Settings:** Remembers each user's API key, model choice, system prompt, and web search preference in a local `database.yaml` file.
*   **Robust Error Handling:** Automatically retries API requests when rate limits are hit and gracefully handles unsupported file types.

## How to Run

## 1. As docker compose:

```yaml
services:
  domovoi_bot:
    build:
      context: https://github.com/procrastinando/domovoi_bot.git#main
    image: procrastinando/domovoi_bot:latest
    container_name: domovoi_bot
    environment:
      # Define your secrets in an .env file or directly here
      TELEGRAM_TOKEN: ${TELEGRAM_TOKEN}
    volumes:
      - domovoi_bot_data:/app
    restart: always
volumes:
  domovoi_bot_data:
```

## 2. Manual installation

### Prerequisites

*   Python 3.10+
*   [FFmpeg](https://ffmpeg.org/download.html) installed and available in your system's PATH (required for audio conversion).
*   A Groq API Key (from [console.groq.com](https://console.groq.com/keys)).
*   A Telegram Bot Token (from [@BotFather](https://t.me/BotFather) on Telegram).

### Installation and Setup

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create a Virtual Environment**
    It is highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    ```

3.  **Activate the Environment**
    *   **On Windows:**
        ```cmd
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Required Libraries**
    Use the provided `requirements.txt` file to install all necessary packages.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure the Bot**
    Open the Python script (e.g., `bot.py`) and replace the placeholder value for your Telegram Bot Token.
    ```python
    # Find this line in the script
    TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE" 
    ```
    *Note: For production, it is better to use environment variables for security.*

6.  **Run the Bot**
    Launch the bot from your terminal.
    ```bash
    python bot.py
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

| Command           | Description                                    |
| ----------------- | ---------------------------------------------- |
| `/start`          | Initializes the bot and sets up your profile.  |
| `/new_chat`       | Archives the current conversation and starts a new one. |
| `/models`         | Choose from a list of available AI models.     |
| `/web_search`     | Toggle the real-time web search ON or OFF.     |
| `/system_prompt`  | Set a custom personality/instruction for the bot. |
| `/api`            | View or update your Groq API key.              |

Any command not in this list will be automatically deleted from the chat to keep it clean.