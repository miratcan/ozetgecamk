# ozetgecamk

Gemini-powered Telegram bot. It logs channel/group messages to SQLite and summarizes recent chat with `/ozetgecamk`. Photos are also routed through Gemini for a short description.

## Quick start (Docker)
1. Copy env vars and fill them:
   ```bash
   cp .env.example .env
   ```
2. Build & run:
   ```bash
   docker compose up -d --build
   ```
3. Logs live at `./data/chat_logs.db` (mounted volume). Tail output with:
   ```bash
   docker compose logs -f
   ```

## Bot commands
- `/start` basic info
- `/ozetgecamk [amount] [unit]` summarizes recent chat. Defaults to 6 hours. Examples:
  - `/ozetgecamk` (last 6 hours)
  - `/ozetgecamk 30 min`
  - `/ozetgecamk 2 days`

Accepted units: minutes (`min`, `dk`, `dakika`), hours (`hour`, `saat`), days (`day`, `gun`).

## Environment variables
- `TELEGRAM_TOKEN`: Telegram bot token (required)
- `GEMINI_API_KEY`: Gemini API key (required)
- `GEMINI_MODEL`: Model name, default `gemini-1.5-flash-latest` (use this if `gemini-1.5-flash` returns 404)
- `RESPONSE_LANGUAGE`: Language Gemini should respond in (default `turkish`)
- `DEFAULT_SUMMARY_HOURS`: Hours to summarize when no args are given (default `6`)
- `DB_PATH`: SQLite path (default `/app/data/chat_logs.db`)

## Local run (optional)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TELEGRAM_TOKEN=... GEMINI_API_KEY=...
python main.py
```
