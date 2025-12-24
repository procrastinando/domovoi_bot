# Groq Multimodal Telegram Bot

A highly advanced, multi-language Telegram bot powered by **Groq's LPU™ Inference Engine**. This bot acts as a versatile personal AI assistant featuring persistent memory, multimodal capabilities (text, image, audio), smart web search integration (Tavily/Compound), and robust API management.

[![Powered by Groq](https://img.shields.io/badge/Powered%20by-Groq-green?style=flat-square)](https://groq.com/)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://www.python.org/)
[![Multi-language](https://img.shields.io/badge/Language-Multi-orange?style=flat-square)](#-features)

## ✨ Features

*   🌍 **Multi-language Support (9 Languages):**
    *   English, Spanish, Chinese, Russian, French, Italian, Portuguese, Vietnamese, Indonesian.
    *   **Domovoi Persona:** System prompts are localized with cultural nuances and local slang (80% English / 20% Local Slang) for a unique personality.

*   🧠 **Advanced AI Models:**
    *   **Llama 4 Maverick**: Multimodal (Text & Vision).
    *   **Kimi K2**: High-speed text processing.
    *   **GPT-Oss 120B**: Deep reasoning capabilities.
    *   **Groq Compound**: Specialized model for complex tool use.

*   🖼️ **Multimodal Intelligence:**
    *   **Vision:** Analyzes photos using Llama 4 (even if the selected chat model is text-only, it auto-switches for description).
    *   **Audio:** Transcribes voice notes using **Whisper-Large-V3** and responds intelligently.

*   🌐 **Smart Web Search (Two Modes):**
    *   **Tavily Search:** Advanced external search. Uses **Groq Structured Outputs** to generate optimal search queries based on conversation history. Results are injected cleanly into the context.
    *   **Compound Native:** Uses Groq's internal `compound` model tools for integrated search.
    *   **Advanced Config:** Configure search depth and answer inclusion directly from the menu.

*   ⚡ **Smart API Load Balancing & Failover:**
    *   **Dual API Keys:** Supports Main and Fallback keys for both Groq and Tavily.
    *   **Randomized Usage:** Automatically balances requests between keys to avoid rate limits.
    *   **Silent Pruning:** If context limits (413) or rate limits (429) are hit, the bot **silently prunes old messages** and recursively retries without disturbing the user.

*   🛠️ **Dynamic & Clean Interface:**
    *   **Ephemeral Messages:** Configuration menus and status updates automatically delete themselves to keep the chat clean.
    *   **Dynamic Menu:** The `/` command menu updates in real-time to show current settings (Model, Temp, Search Mode, etc.).
    *   **Visual History Management:** `/delete_last` visually removes the last interaction from Telegram AND the database.

*   📄 **Telegra.ph Integration:**
    *   Automatically posts long responses to Telegra.ph for better readability (configurable).

*   💾 **Persistent Privacy:**
    *   User settings and chat history are saved locally in `database.yaml`.
    *   `/erase_me`: Instantly wipes all user data from the bot.

## How to Run

### 1. Using Docker Compose (Recommended)

1.  **Configure Environment:**
    Create a `.env` file:
    ```env
    TELEGRAM_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
    ```

2.  **Files:**
    Ensure `languages.yaml` is in the same directory.

3.  **Run:**
    ```bash
    docker-compose up -d
    ```

### 2. Manual Installation

**Prerequisites:**
*   Python 3.10+
*   **FFmpeg** installed (required for voice processing).
*   **Groq API Key** ([Get it here](https://console.groq.com/keys)).
*   **Tavily API Key** (Optional, for web search).

**Steps:**
1.  Clone the repo and navigate to the directory.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set Token:
    ```bash
    export TELEGRAM_TOKEN="YOUR_TOKEN"
    # Windows: set TELEGRAM_TOKEN="YOUR_TOKEN"
    ```
4.  Run:
    ```bash
    python domobot.py
    ```

## Usage & Commands

Start the bot with `/start`. Use the **Menu button** or type commands to configure behavior.

| Command | Description |
| :--- | :--- |
| `/new_chat` | 🧹 Clears context history. Shows Tavily usage stats if configured. |
| `/delete_last` | 🗑️ Deletes the last user-bot interaction from chat and memory. |
| `/models` | 🤖 Switch between Llama 4, Kimi, GPT-Oss, or Compound. |
| `/web_search` | 🌐 Toggle between **Off**, **Compound**, or **Tavily** modes. |
| `/prompt` | ⚙️ Configure System, Image, and Search Query generation prompts. |
| `/api` | 🔑 Manage Main/Fallback keys for Groq and Tavily. |
| `/temperature` | 🌡️ Adjust creativity (0.0 - 2.0). |
| `/max_completion_tokens` | 📏 Set max response length (2k - 65k). Auto-clamps to model limits. |
| `/use_telegraph` | 📄 Configure long message handling (Never/Long/Always). |
| `/language` | 🌍 Change bot interface language. |
| `/erase_me` | ⚠️ Permanently delete all your data. |

### Prompt Copying
When viewing current prompts or API keys, tap the text (e.g., `gsk_...`) to instantly copy it to your clipboard.

### Context Management
*   **Web Search Results** are ephemeral. They are injected into the context for the *current* answer but **are not saved** to the database to save tokens.
*   **Silent Failover**: If you see the bot "typing" for a bit longer than usual, it is likely handling a rate limit by optimizing context and retrying in the background.