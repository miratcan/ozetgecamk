import io
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Tuple

import google.generativeai as genai
from PIL import Image
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
RESPONSE_LANGUAGE = os.getenv("RESPONSE_LANGUAGE", "turkish")
DEFAULT_SUMMARY_HOURS = int(os.getenv("DEFAULT_SUMMARY_HOURS", "6"))
DB_PATH = Path(os.getenv("DB_PATH", "/app/data/chat_logs.db"))
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", "10"))
RATE_LIMIT_IMAGE = Path(__file__).parent / "rate_limit.jpg"
_model = None
_last_summary_time: dict[str, datetime] = {}  # chat_id -> last request time


def init_db() -> None:
    """Create SQLite database and table if missing."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT,
                user_name TEXT,
                message TEXT,
                timestamp DATETIME
            )
            """
        )
        conn.commit()


def save_log(chat_id: str, user_name: str, message: str) -> None:
    """Persist a message into SQLite."""

    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO logs (chat_id, user_name, message, timestamp) VALUES (?, ?, ?, ?)",
            (str(chat_id), user_name, message, timestamp),
        )
        conn.commit()


def get_logs(chat_id: str, time_delta: timedelta) -> Iterable[Tuple[str, str, str]]:
    """Return logs within the given window."""

    search_time = datetime.utcnow() - time_delta
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT user_name, message, timestamp
            FROM logs
            WHERE chat_id = ? AND timestamp > ?
            ORDER BY timestamp ASC
            """,
            (str(chat_id), search_time.isoformat(timespec="seconds")),
        ).fetchall()
    return rows


def parse_duration(args: list[str]) -> Tuple[timedelta, str]:
    """Parse duration args (e.g., "3 hours", "30 min")."""

    if not args:
        return timedelta(hours=DEFAULT_SUMMARY_HOURS), f"{DEFAULT_SUMMARY_HOURS} hours"

    try:
        amount = int(args[0])
    except ValueError:
        return (
            timedelta(hours=DEFAULT_SUMMARY_HOURS),
            f"{DEFAULT_SUMMARY_HOURS} hours (default; invalid input)",
        )

    unit = args[1].lower() if len(args) > 1 else "hours"

    minute_tokens = ("min", "mins", "minute", "minutes", "dk", "dak", "dakika")
    hour_tokens = ("hour", "hours", "hr", "hrs", "saat")
    day_tokens = ("day", "days", "gun")

    if unit.startswith(minute_tokens):
        return timedelta(minutes=amount), f"{amount} minutes"
    if unit.startswith(day_tokens):
        return timedelta(days=amount), f"{amount} days"
    if unit.startswith(hour_tokens):
        return timedelta(hours=amount), f"{amount} hours"

    return timedelta(hours=DEFAULT_SUMMARY_HOURS), f"{DEFAULT_SUMMARY_HOURS} hours (default; unknown unit)"


def ensure_gemini_model():
    global _model
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    if _model is None:
        genai.configure(api_key=GEMINI_API_KEY)

        try:
            _model = genai.GenerativeModel(GEMINI_MODEL)
        except Exception as exc:
            # Some accounts expose only "*-latest" or older names; try a fallback.
            if "404" in str(exc) and not GEMINI_MODEL.endswith("-latest"):
                fallback = f"{GEMINI_MODEL}-latest"
                logger.warning(
                    "Model %s not found; falling back to %s", GEMINI_MODEL, fallback
                )
                _model = genai.GenerativeModel(fallback)
            else:
                raise

    return _model


async def analyze_image(image_bytes: bytes) -> str:
    """Describe an image via Gemini."""

    model = ensure_gemini_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    prompt = (
        "Describe what is in the image. If it is a screenshot, read any visible text. "
        f"Be brief and clear. Respond in {RESPONSE_LANGUAGE}."
    )

    response = model.generate_content(
        [prompt, {"mime_type": "image/png", "data": png_bytes}]
    )
    return response.text.strip()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    chat_id = update.effective_chat.id if update.effective_chat else "unknown"

    if not message:
        return

    if message.from_user:
        user = (
            message.from_user.full_name
            or message.from_user.username
            or "Unknown"
        )
    else:
        user = "Unknown"

    try:
        if message.text and not message.text.startswith("/"):
            save_log(chat_id, user, message.text)
            logger.info("[%s] %s", user, message.text)
            return

        if message.photo:
            photo_file = await message.photo[-1].get_file()
            image_bytes = await photo_file.download_as_bytearray()
            description = await analyze_image(image_bytes)
            caption = message.caption or ""
            log_entry = f"[IMAGE ANALYSIS] {description}"
            if caption:
                log_entry = f"{log_entry} | Caption: {caption}"
            save_log(chat_id, user, log_entry)
            logger.info("[%s] %s", user, log_entry)
            return

        if message.caption:
            save_log(chat_id, user, f"[MEDIA CAPTION] {message.caption}")
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.exception("Error while handling message: %s", exc)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "Hi! I am logging messages.\n"
            "Use /ozetgecamk to summarize recent chat (e.g., /ozetgecamk 3 hours)."
        )


async def summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return

    chat_id = str(update.effective_chat.id) if update.effective_chat else "unknown"
    now = datetime.utcnow()

    # Rate limit check
    if chat_id in _last_summary_time:
        elapsed = (now - _last_summary_time[chat_id]).total_seconds()
        if elapsed < RATE_LIMIT_SECONDS:
            await update.message.reply_photo(
                photo=open(RATE_LIMIT_IMAGE, "rb"),
                caption=f"Yavaş ol! {RATE_LIMIT_SECONDS - int(elapsed)} saniye bekle."
            )
            return

    _last_summary_time[chat_id] = now

    duration, label = parse_duration(context.args)
    status = await update.message.reply_text(f"Reviewing the last {label}...")

    rows = get_logs(chat_id, duration)
    if not rows:
        await status.edit_text(f"No logs found for the last {label}.")
        return

    transcript = "\n".join(
        f"{row['timestamp']} - {row['user_name']}: {row['message']}" for row in rows
    )

    prompt = (
        f"The following logs are from a Telegram chat over the last {label}.\n"
        "Summarize the conversation with a witty, gossip-like tone.\n"
        "Skip fluff; focus on key events and who said what.\n"
        "Format rules:\n"
        "- Use *bold* for user names (single asterisk)\n"
        "- Use bullet points (•) for each topic\n"
        "- Group related messages under topic headers\n"
        f"Write the summary in {RESPONSE_LANGUAGE}.\n\n"
        f"Logs:\n{transcript}"
    )

    try:
        model = ensure_gemini_model()
        response = model.generate_content(prompt)
        await status.edit_text(response.text[:4000], parse_mode="Markdown")
    except Exception as exc:  # pragma: no cover - runtime safety
        logger.exception("Summarization error: %s", exc)
        await status.edit_text(f"Error: {exc}")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Telegram error: %s", context.error)


def main() -> None:
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN is missing.")

    init_db()

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("ozetgecamk", summarize))
    application.add_handler(
        MessageHandler(filters.ALL & ~filters.COMMAND, handle_message)
    )
    application.add_error_handler(error_handler)

    logger.info("Bot is live and listening...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
